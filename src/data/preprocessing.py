"""
Data Preprocessing and Pipeline Module for RecoSphere AI.
Performs data cleaning, chronological train-val-test splitting,
and reproducible ID encoding without data leakage.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from configs.settings import PROCESSED_DATA_DIR, DataConfig
from src.data.validation import DataValidator


class DataPreprocessor:
    """Preprocesses interaction events and catalog metadata for recommendation modeling."""

    def __init__(self, config: Optional[DataConfig] = None, output_dir: Path = PROCESSED_DATA_DIR):
        self.config = config or DataConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.category_to_idx: Dict[int, int] = {}
        self.idx_to_category: Dict[int, int] = {}

    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates, sorts chronologically, and applies event weighting."""
        df_clean = df.copy()

        # Deduplicate identical events at identical timestamps
        df_clean = df_clean.drop_duplicates(subset=["timestamp", "visitorid", "itemid", "event"])

        # Ensure timestamp is numeric and sort
        df_clean["timestamp"] = pd.to_numeric(df_clean["timestamp"], errors="coerce")
        df_clean = df_clean.dropna(subset=["timestamp", "visitorid", "itemid", "event"])
        df_clean["timestamp"] = df_clean["timestamp"].astype(np.int64)
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

        # Map event type to interaction weight
        weight_map = self.config.event_weights
        df_clean["weight"] = df_clean["event"].map(lambda e: weight_map.get(e, 1.0)).astype(np.float32)

        return df_clean

    def split_chronological(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronologically splits events into train, val, test sets using time cutoffs.
        Guarantees no future temporal leakage into training data.
        """
        n = len(df)
        train_end_idx = int(n * self.config.train_ratio)
        val_end_idx = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end_idx].copy()
        val_df = df.iloc[train_end_idx:val_end_idx].copy()
        test_df = df.iloc[val_end_idx:].copy()

        return train_df, val_df, test_df

    def fit_encoders(self, train_events: pd.DataFrame, catalog: pd.DataFrame):
        """
        Fits ID encoders on training interactions and catalog items.
        Known items and users are assigned contiguous indexes.
        """
        # User encoder: known users in training set
        unique_users = sorted(train_events["visitorid"].unique())
        self.user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: int(uid) for uid, idx in self.user_to_idx.items()}

        # Item encoder: includes all catalog items
        catalog_items = sorted(catalog["itemid"].unique())
        self.item_to_idx = {int(iid): idx for idx, iid in enumerate(catalog_items)}
        self.idx_to_item = {idx: int(iid) for iid, idx in self.item_to_idx.items()}

        # Category encoder
        if "categoryid" in catalog.columns:
            unique_cats = sorted(catalog["categoryid"].dropna().unique())
            self.category_to_idx = {int(cid): idx for idx, cid in enumerate(unique_cats)}
            self.idx_to_category = {idx: int(cid) for cid, idx in self.category_to_idx.items()}

    def transform_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes user and item IDs into contiguous integer indices. Unseen entities become -1."""
        df_enc = df.copy()
        df_enc["user_idx"] = df_enc["visitorid"].map(self.user_to_idx).fillna(-1).astype(np.int32)
        df_enc["item_idx"] = df_enc["itemid"].map(self.item_to_idx).fillna(-1).astype(np.int32)
        return df_enc

    def run_pipeline(
        self, raw_events: pd.DataFrame, raw_catalog: pd.DataFrame
    ) -> Dict[str, Any]:
        """Executes full preprocessing pipeline and saves processed artifacts."""
        print("[DataPreprocessor] Validating raw events and catalog...")
        is_events_valid, events_report = DataValidator.validate_events(raw_events)
        is_cat_valid, cat_report = DataValidator.validate_catalog(raw_catalog)

        print("[DataPreprocessor] Cleaning and sorting interaction events chronologically...")
        clean_df = self.clean_events(raw_events)

        print("[DataPreprocessor] Splitting chronologically (train/val/test)...")
        train_raw, val_raw, test_raw = self.split_chronological(clean_df)

        print("[DataPreprocessor] Fitting ID encoders...")
        self.fit_encoders(train_raw, raw_catalog)

        print("[DataPreprocessor] Transforming splits...")
        train_df = self.transform_events(train_raw)
        val_df = self.transform_events(val_raw)
        test_df = self.transform_events(test_raw)

        # Enrich catalog with category_idx and save
        catalog_df = raw_catalog.copy()
        if "categoryid" in catalog_df.columns:
            catalog_df["category_idx"] = catalog_df["categoryid"].map(self.category_to_idx).fillna(-1).astype(np.int32)
        catalog_df["item_idx"] = catalog_df["itemid"].map(self.item_to_idx).fillna(-1).astype(np.int32)

        # Save processed data
        train_df.to_csv(self.output_dir / "train_events.csv", index=False)
        val_df.to_csv(self.output_dir / "val_events.csv", index=False)
        test_df.to_csv(self.output_dir / "test_events.csv", index=False)
        catalog_df.to_csv(self.output_dir / "catalog.csv", index=False)

        # Calculate statistics
        stats = {
            "validation_report": events_report,
            "catalog_validation": cat_report,
            "num_users": len(self.user_to_idx),
            "num_items": len(self.item_to_idx),
            "num_categories": len(self.category_to_idx),
            "train_events_count": len(train_df),
            "val_events_count": len(val_df),
            "test_events_count": len(test_df),
            "val_cold_users_pct": float((val_df["user_idx"] == -1).mean() * 100),
            "test_cold_users_pct": float((test_df["user_idx"] == -1).mean() * 100),
            "split_ratios": {
                "train": float(len(train_df) / len(clean_df)),
                "val": float(len(val_df) / len(clean_df)),
                "test": float(len(test_df) / len(clean_df)),
            },
        }

        # Save metadata and encoders
        meta = {
            "stats": stats,
            "user_to_idx": {str(k): v for k, v in self.user_to_idx.items()},
            "item_to_idx": {str(k): v for k, v in self.item_to_idx.items()},
            "category_to_idx": {str(k): v for k, v in self.category_to_idx.items()},
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[DataPreprocessor] Pipeline completed successfully. Artifacts saved in {self.output_dir}")
        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "catalog_df": catalog_df,
            "stats": stats,
        }

    def load_processed(self) -> Dict[str, Any]:
        """Loads already processed artifacts from disk."""
        train_df = pd.read_csv(self.output_dir / "train_events.csv")
        val_df = pd.read_csv(self.output_dir / "val_events.csv")
        test_df = pd.read_csv(self.output_dir / "test_events.csv")
        catalog_df = pd.read_csv(self.output_dir / "catalog.csv")

        with open(self.output_dir / "metadata.json", "r") as f:
            meta = json.load(f)

        self.user_to_idx = {int(k): v for k, v in meta["user_to_idx"].items()}
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.item_to_idx = {int(k): v for k, v in meta["item_to_idx"].items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        self.category_to_idx = {int(k): v for k, v in meta["category_to_idx"].items()}
        self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "catalog_df": catalog_df,
            "stats": meta["stats"],
        }

