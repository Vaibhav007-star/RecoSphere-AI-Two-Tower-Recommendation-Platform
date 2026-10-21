"""
Automated Unit Tests for Two-Tower Neural Network Architecture, Loss, and Checkpoints.
"""

import pytest
import torch
import torch.nn.functional as F

from configs.settings import ModelConfig
from src.models.two_tower import TwoTowerModel, UserTower, ItemTower
from src.models.losses import InBatchSampledSoftmaxLoss


def test_user_tower_forward():
    num_users = 100
    config = ModelConfig(embedding_dim=32, hidden_dims=[64, 32])
    tower = UserTower(num_users=num_users, embedding_dim=32, hidden_dims=[64, 32])

    user_indices = torch.tensor([0, 1, 5, 99], dtype=torch.long)
    features = torch.randn(4, 2)

    emb = tower(user_indices, features)
    assert emb.shape == (4, 32)
    # Check L2 normalization
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_item_tower_forward():
    num_items = 200
    num_categories = 25
    tower = ItemTower(num_items=num_items, num_categories=num_categories, embedding_dim=32, hidden_dims=[64, 32])

    item_indices = torch.tensor([0, 10, 50, 199], dtype=torch.long)
    cat_indices = torch.tensor([1, 2, 5, 20], dtype=torch.long)
    item_feats = torch.randn(4, 2)

    emb = tower(item_indices, cat_indices, item_feats)
    assert emb.shape == (4, 32)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_two_tower_similarity_and_loss():
    config = ModelConfig(embedding_dim=16, hidden_dims=[32, 16], temperature=0.1)
    model = TwoTowerModel(num_users=50, num_items=50, num_categories=10, config=config)

    batch_size = 8
    u_idx = torch.randint(0, 50, (batch_size,))
    i_idx = torch.randint(0, 50, (batch_size,))

    u_emb, i_emb = model(u_idx, i_idx)
    assert u_emb.shape == (batch_size, 16)
    assert i_emb.shape == (batch_size, 16)

    criterion = InBatchSampledSoftmaxLoss(temperature=config.temperature)
    loss = criterion(u_emb, i_emb)

    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

