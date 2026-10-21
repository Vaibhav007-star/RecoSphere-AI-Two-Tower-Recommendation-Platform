"""
Candidate Retrieval Vector Index for RecoSphere AI.
Supports fast inner-product and cosine similarity search over persistent item embeddings
with fallback to PyTorch tensor matrix multiplication.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Set
import numpy as np
import torch

from configs.settings import INDEX_DIR


class VectorIndex:
    """Vector index for item candidate retrieval in embedding space."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.item_ids: List[int] = []
        self.item_embeddings: Optional[torch.Tensor] = None
        self.id_to_index: dict = {}

    def build_index(self, item_ids: List[int], embeddings: torch.Tensor):
        """
        Builds the retrieval index from item IDs and their dense L2-normalized embeddings.
        embeddings: [num_items, dim]
        """
        self.item_ids = [int(i) for i in item_ids]
        # Ensure tensor is CPU and float32 normalized
        self.item_embeddings = embeddings.cpu().float()
        self.id_to_index = {iid: idx for idx, iid in enumerate(self.item_ids)}
        print(f"[VectorIndex] Built index with {len(self.item_ids)} item vectors (dim={self.item_embeddings.size(1)})")

    def search(
        self,
        query_vector: torch.Tensor,
        top_k: int = 100,
        exclude_items: Optional[Set[int]] = None,
        category_filter: Optional[Set[int]] = None,
        item_to_category: Optional[dict] = None,
    ) -> List[Tuple[int, float]]:
        """
        Searches top_k most similar items to query_vector (cosine / dot-product).
        Returns list of (item_id, score) tuples.
        """
        if self.item_embeddings is None or len(self.item_ids) == 0:
            return []

        exclude = exclude_items or set()
        q = query_vector.cpu().float().view(1, -1)

        # Batch dot product (cosine similarity since vectors are L2-normalized)
        scores = torch.mm(q, self.item_embeddings.T).squeeze(0)  # [num_items]

        # Apply exclusion / filters
        if exclude or category_filter:
            for idx, iid in enumerate(self.item_ids):
                if iid in exclude:
                    scores[idx] = -float("inf")
                elif category_filter and item_to_category:
                    cat = item_to_category.get(iid)
                    if cat not in category_filter:
                        scores[idx] = -float("inf")

        # Top-K retrieval
        k = min(top_k, len(self.item_ids))
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if s > -float("inf"):
                results.append((self.item_ids[idx], float(s)))

        return results

    def search_similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        category_filter: Optional[Set[int]] = None,
        item_to_category: Optional[dict] = None,
    ) -> List[Tuple[int, float]]:
        """
        Finds the top-K nearest neighbor products to a given item_id in embedding space.
        """
        idx = self.id_to_index.get(item_id)
        if idx is None or self.item_embeddings is None:
            return []

        item_vec = self.item_embeddings[idx]
        return self.search(
            query_vector=item_vec,
            top_k=top_k,
            exclude_items={item_id},
            category_filter=category_filter,
            item_to_category=item_to_category,
        )

    def save(self, name: str = "item_index.pt"):
        """Persists the index to disk."""
        save_path = self.index_dir / name
        torch.save(
            {
                "item_ids": self.item_ids,
                "embeddings": self.item_embeddings,
            },
            save_path,
        )
        print(f"[VectorIndex] Index saved to {save_path}")

    def load(self, name: str = "item_index.pt") -> bool:
        """Loads index from disk."""
        load_path = self.index_dir / name
        if not load_path.exists():
            return False
        data = torch.load(load_path, map_location="cpu")
        self.item_ids = data["item_ids"]
        self.item_embeddings = data["embeddings"]
        self.id_to_index = {iid: idx for idx, iid in enumerate(self.item_ids)}
        print(f"[VectorIndex] Loaded index with {len(self.item_ids)} items from {load_path}")
        return True

