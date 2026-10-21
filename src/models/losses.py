"""
Loss Functions for Two-Tower Recommendation Retrieval.
Implements:
1. In-Batch Sampled Softmax Contrastive Loss (InfoNCE)
2. Weighted Contrastive Loss (accounting for interaction type: view, cart, purchase)
3. Cosine Margin Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InBatchSampledSoftmaxLoss(nn.Module):
    """
    In-batch sampled softmax loss (InfoNCE).
    Efficient retrieval objective that treats other items in the mini-batch as negative candidates.
    Supports sample interaction weights (e.g., transaction > addtocart > view).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        user_embeddings: [batch_size, dim] (L2 normalized)
        item_embeddings: [batch_size, dim] (L2 normalized)
        weights: [batch_size] optional float weights
        """
        # Similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature

        # Ground truth: positive item for user i is item i (diagonal)
        batch_size = user_embeddings.size(0)
        labels = torch.arange(batch_size, device=user_embeddings.device)

        if weights is not None:
            # Weighted cross-entropy across batch
            loss_per_sample = F.cross_entropy(similarity_matrix, labels, reduction="none")
            weighted_loss = (loss_per_sample * weights).sum() / (weights.sum() + 1e-8)
            return weighted_loss
        else:
            return F.cross_entropy(similarity_matrix, labels)


class CosineMarginLoss(nn.Module):
    """Pairwise cosine margin ranking loss for user-item retrieval."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        user_embeddings: torch.Tensor,
        pos_item_embeddings: torch.Tensor,
        neg_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = F.cosine_similarity(user_embeddings, pos_item_embeddings)
        neg_sim = F.cosine_similarity(user_embeddings, neg_item_embeddings)
        loss = F.relu(self.margin - pos_sim + neg_sim)
        return loss.mean()

