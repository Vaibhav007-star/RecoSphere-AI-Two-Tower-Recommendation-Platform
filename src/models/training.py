"""
Two-Tower Model Training Pipeline for RecoSphere AI.
Implements reproducible CPU-friendly mini-batch training with in-batch negatives,
vectorized feature generation, early stopping, checkpointing, and metric logging.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from configs.settings import MODELS_DIR, ModelConfig
from src.models.two_tower import TwoTowerModel
from src.models.losses import InBatchSampledSoftmaxLoss


class InteractionDataset(Dataset):
    """Vectorized, high-performance PyTorch Dataset for e-commerce interaction pairs."""

    def __init__(
        self,
        events_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        num_categories: int,
    ):
        # Filter to events where user and item are mapped (>= 0)
        valid = events_df[(events_df["user_idx"] >= 0) & (events_df["item_idx"] >= 0)].copy()

        self.user_indices = torch.tensor(valid["user_idx"].values, dtype=torch.long)
        self.item_indices = torch.tensor(valid["item_idx"].values, dtype=torch.long)

        # Weights
        if "weight" in valid.columns:
            self.weights = torch.tensor(valid["weight"].values, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(valid), dtype=torch.float32)

        # Vectorized category mapping
        if "category_idx" in catalog_df.columns:
            item_to_cat = dict(zip(catalog_df["item_idx"].values, catalog_df["category_idx"].values))
            cats = valid["item_idx"].map(item_to_cat).fillna(num_categories).astype(np.int64).values
            self.category_indices = torch.tensor(cats, dtype=torch.long)
        else:
            self.category_indices = torch.full((len(valid),), num_categories, dtype=torch.long)

        # Vectorized continuous item features: [price, available]
        if "price" in catalog_df.columns:
            price_map = dict(zip(catalog_df["item_idx"].values, np.log1p(catalog_df["price"].fillna(20.0).values)))
            prices = valid["item_idx"].map(price_map).fillna(2.5).astype(np.float32).values
        else:
            prices = np.full(len(valid), 2.5, dtype=np.float32)

        if "available" in catalog_df.columns:
            avail_map = dict(zip(catalog_df["item_idx"].values, catalog_df["available"].fillna(1.0).values))
            avails = valid["item_idx"].map(avail_map).fillna(1.0).astype(np.float32).values
        else:
            avails = np.full(len(valid), 1.0, dtype=np.float32)

        self.item_features = torch.tensor(np.column_stack([prices, avails]), dtype=torch.float32)

        # Vectorized continuous user activity features: [log_user_event_count, avg_weight]
        user_counts = valid["user_idx"].value_counts().to_dict()
        user_cnt_arr = valid["user_idx"].map(user_counts).fillna(1.0).values
        log_cnt = np.log1p(user_cnt_arr).astype(np.float32)
        dummy_weights = np.ones(len(valid), dtype=np.float32)

        self.user_features = torch.tensor(np.column_stack([log_cnt, dummy_weights]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.weights[idx],
            self.category_indices[idx],
            self.item_features[idx],
            self.user_features[idx],
        )


class TwoTowerTrainer:
    """Orchestrates model training, validation, early stopping, and artifact persistence."""

    def __init__(self, config: Optional[ModelConfig] = None, output_dir: Path = MODELS_DIR):
        self.config = config or ModelConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.config.device)

        # Reproducibility
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        num_categories: int,
    ) -> Tuple[TwoTowerModel, Dict[str, Any]]:
        """Trains the Two-Tower model on training interactions and tunes on validation set."""
        print(f"[TwoTowerTrainer] Initializing Two-Tower Neural Network (Users: {num_users}, Items: {num_items}, Categories: {num_categories})", flush=True)

        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            num_categories=num_categories,
            config=self.config,
        ).to(self.device)

        train_dataset = InteractionDataset(train_df, catalog_df, num_users, num_items, num_categories)
        val_dataset = InteractionDataset(val_df, catalog_df, num_users, num_items, num_categories)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        criterion = InBatchSampledSoftmaxLoss(temperature=self.config.temperature)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_reg,
        )

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "epoch_duration_sec": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = self.output_dir / "best_two_tower_model.pt"

        print(f"[TwoTowerTrainer] Starting training for {self.config.epochs} epochs (Batch size: {self.config.batch_size})...", flush=True)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            model.train()
            train_losses = []

            for u_idx, i_idx, weights, c_idx, i_feats, u_feats in train_loader:
                u_idx = u_idx.to(self.device)
                i_idx = i_idx.to(self.device)
                weights = weights.to(self.device)
                c_idx = c_idx.to(self.device)
                i_feats = i_feats.to(self.device)
                u_feats = u_feats.to(self.device)

                optimizer.zero_grad()
                u_emb, i_emb = model(u_idx, i_idx, u_feats, c_idx, i_feats)
                loss = criterion(u_emb, i_emb, weights=weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation pass
            model.eval()
            val_losses = []
            with torch.no_grad():
                for u_idx, i_idx, weights, c_idx, i_feats, u_feats in val_loader:
                    if len(u_idx) <= 1:
                        continue
                    u_idx = u_idx.to(self.device)
                    i_idx = i_idx.to(self.device)
                    weights = weights.to(self.device)
                    c_idx = c_idx.to(self.device)
                    i_feats = i_feats.to(self.device)
                    u_feats = u_feats.to(self.device)

                    u_emb, i_emb = model(u_idx, i_idx, u_feats, c_idx, i_feats)
                    v_loss = criterion(u_emb, i_emb, weights=weights)
                    val_losses.append(v_loss.item())

            mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss
            duration = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(round(mean_train_loss, 4))
            history["val_loss"].append(round(mean_val_loss, 4))
            history["epoch_duration_sec"].append(round(duration, 2))

            print(f"[Epoch {epoch:02d}/{self.config.epochs:02d}] Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} ({duration:.2f}s)", flush=True)

            # Checkpoint on best validation loss
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(self.config),
                        "num_users": num_users,
                        "num_items": num_items,
                        "num_categories": num_categories,
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                    },
                    best_model_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"[TwoTowerTrainer] Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.4f})", flush=True)
                    break

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"[TwoTowerTrainer] Model checkpoint and history saved at {self.output_dir}", flush=True)
        return model, history

    @classmethod
    def load_model(cls, model_path: Path, device: str = "cpu") -> TwoTowerModel:
        """Loads trained model checkpoint from disk."""
        checkpoint = torch.load(model_path, map_location=device)
        model = TwoTowerModel(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            num_categories=checkpoint["num_categories"],
            config=ModelConfig(**checkpoint.get("config", {})),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

