"""
Two-Tower Deep Neural Network for Personalized E-Commerce Recommendation.
Independent User Tower and Item Tower generating dense L2-normalized representations
in a shared metric space.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.settings import ModelConfig


class UserTower(nn.Module):
    """
    User Tower: Embeds user identity, historical engagement frequency, and preference signals
    into a dense normalized user representation.
    """

    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        # +1 to handle unknown/padding users (idx = num_users)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # Input dimension: user embedding + 2 continuous user activity stats (log frequency, avg weight)
        in_dim = embedding_dim + 2

        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        # Final projection to shared retrieval embedding dimension
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, user_indices: torch.Tensor, user_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        user_indices: [batch_size] int tensor
        user_features: [batch_size, 2] continuous features (or None)
        Returns: [batch_size, output_dim] L2-normalized tensor
        """
        u_emb = self.user_embedding(user_indices)

        if user_features is not None:
            combined = torch.cat([u_emb, user_features], dim=-1)
        else:
            dummy_feats = torch.zeros((user_indices.size(0), 2), device=user_indices.device)
            combined = torch.cat([u_emb, dummy_feats], dim=-1)

        projected = self.mlp(combined)
        return F.normalize(projected, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Item Tower: Embeds item identity, categorical taxonomy, and price/availability features
    into the shared metric space.
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int = 50,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        cat_dim = max(8, embedding_dim // 4)

        # +1 for unknown/padding items & categories
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=num_items)
        self.category_embedding = nn.Embedding(num_categories + 1, cat_dim, padding_idx=num_categories)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)

        # Input dimension: item_emb + cat_emb + 2 numerical features (normalized log price, availability)
        in_dim = embedding_dim + cat_dim + 2

        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(
        self,
        item_indices: torch.Tensor,
        category_indices: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        item_indices: [batch_size] int tensor
        category_indices: [batch_size] int tensor (or None)
        item_features: [batch_size, 2] continuous features (or None)
        Returns: [batch_size, output_dim] L2-normalized tensor
        """
        i_emb = self.item_embedding(item_indices)

        if category_indices is not None:
            c_emb = self.category_embedding(category_indices)
        else:
            c_emb = torch.zeros((item_indices.size(0), self.category_embedding.embedding_dim), device=item_indices.device)

        if item_features is not None:
            combined = torch.cat([i_emb, c_emb, item_features], dim=-1)
        else:
            dummy_feats = torch.zeros((item_indices.size(0), 2), device=item_indices.device)
            combined = torch.cat([i_emb, c_emb, dummy_feats], dim=-1)

        projected = self.mlp(combined)
        return F.normalize(projected, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower Recommendation Model orchestrating User Tower and Item Tower.
    Allows independent embedding extraction for scalable vector search.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int = 50,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.num_users = num_users
        self.num_items = num_items
        self.num_categories = num_categories

        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        )

        self.item_tower = ItemTower(
            num_items=num_items,
            num_categories=num_categories,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        category_indices: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
    ):
        """Forward pass for mini-batch training. Returns user and item dense normalized embeddings."""
        u_emb = self.user_tower(user_indices, user_features)
        i_emb = self.item_tower(item_indices, category_indices, item_features)
        return u_emb, i_emb

    def get_user_embedding(
        self, user_idx: int, user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference helper: generates a single user embedding."""
        self.eval()
        with torch.no_grad():
            idx_tensor = torch.tensor([user_idx], dtype=torch.long)
            return self.user_tower(idx_tensor, user_features).squeeze(0)

    def get_all_item_embeddings(
        self,
        all_item_indices: torch.Tensor,
        all_category_indices: Optional[torch.Tensor] = None,
        all_item_features: Optional[torch.Tensor] = None,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """Precomputes item embeddings across entire catalog in mini-batches."""
        self.eval()
        embeddings = []
        n = all_item_indices.size(0)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                sub_items = all_item_indices[start:end]
                sub_cats = all_category_indices[start:end] if all_category_indices is not None else None
                sub_feats = all_item_features[start:end] if all_item_features is not None else None

                sub_emb = self.item_tower(sub_items, sub_cats, sub_feats)
                embeddings.append(sub_emb.cpu())

        return torch.cat(embeddings, dim=0)

