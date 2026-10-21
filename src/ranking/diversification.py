"""
Diversification and Fairness Module for RecoSphere AI.
Implements Maximal Marginal Relevance (MMR) and Category Balancing
to avoid echo chambers and single-category over-concentration.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np


class MMRDiversifier:
    """
    Maximal Marginal Relevance (MMR) Diversifier.
    Balances relevance score with diversity against already selected items.
    """

    def __init__(self, lmbda: float = 0.7):
        """
        lmbda: 1.0 means pure relevance, 0.0 means maximum diversity.
        """
        self.lmbda = lmbda

    def diversify(
        self,
        candidate_items: List[Tuple[int, float]],
        item_to_category: Dict[int, int],
        top_n: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Applies MMR diversification using category overlap as similarity penalty.
        candidate_items: list of (item_id, relevance_score)
        """
        if not candidate_items:
            return []

        selected: List[Tuple[int, float]] = []
        selected_cats: List[int] = []
        remaining = list(candidate_items)

        # Pick the highest ranked item first
        first_item = remaining.pop(0)
        selected.append(first_item)
        if first_item[0] in item_to_category:
            selected_cats.append(item_to_category[first_item[0]])

        while len(selected) < top_n and remaining:
            best_idx = -1
            best_mmr_score = -float("inf")

            for idx, (itm, score) in enumerate(remaining):
                cat = item_to_category.get(itm)
                # Penalty based on how many times this category has already been selected
                cat_count = selected_cats.count(cat) if cat is not None else 0
                sim_to_selected = min(1.0, cat_count * 0.4)

                mmr_score = self.lmbda * score - (1.0 - self.lmbda) * sim_to_selected

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                chosen = remaining.pop(best_idx)
                selected.append(chosen)
                if chosen[0] in item_to_category:
                    selected_cats.append(item_to_category[chosen[0]])
            else:
                break

        return selected

