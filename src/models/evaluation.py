"""
Recommendation Evaluation Engine for RecoSphere AI.
Calculates standard Top-K ranking, retrieval, coverage, and diversity metrics:
- Precision@K
- Recall@K
- HitRate@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Catalog Coverage
- Intra-List Diversity / Category Coverage
- Popularity Bias
"""

from typing import Dict, List, Set, Any, Optional
import math
import numpy as np
import pandas as pd


class RecommendationEvaluator:
    """Computes ranking accuracy, catalog coverage, and diversity across a test interaction set."""

    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = k_values or [5, 10, 20]

    def evaluate(
        self,
        recommendations_by_user: Dict[int, List[int]],
        ground_truth_by_user: Dict[int, Set[int]],
        all_catalog_items: Set[int],
        item_to_category: Optional[Dict[int, int]] = None,
        item_popularity_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation across all users who have at least one ground-truth test item."""
        eval_users = [u for u in ground_truth_by_user if u in recommendations_by_user and len(ground_truth_by_user[u]) > 0]
        n_users = len(eval_users)

        if n_users == 0:
            return {"error": "No overlapping users between recommendations and ground truth."}

        results: Dict[str, Any] = {"num_evaluated_users": n_users}

        for k in self.k_values:
            precisions = []
            recalls = []
            hit_rates = []
            ndcgs = []
            mrrs = []
            all_recommended_items_at_k: Set[int] = set()
            intra_list_diversities = []
            pop_scores = []

            for u in eval_users:
                recs = recommendations_by_user[u][:k]
                actuals = ground_truth_by_user[u]
                all_recommended_items_at_k.update(recs)

                hits = [1 if item in actuals else 0 for item in recs]
                num_hits = sum(hits)

                precisions.append(num_hits / k if k > 0 else 0.0)
                recalls.append(num_hits / len(actuals) if len(actuals) > 0 else 0.0)
                hit_rates.append(1.0 if num_hits > 0 else 0.0)

                dcg = sum((hit / math.log2(idx + 2)) for idx, hit in enumerate(hits))
                ideal_hits = min(len(actuals), k)
                idcg = sum((1.0 / math.log2(idx + 2)) for idx in range(ideal_hits))
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

                first_hit_idx = next((idx for idx, hit in enumerate(hits) if hit == 1), None)
                mrrs.append(1.0 / (first_hit_idx + 1) if first_hit_idx is not None else 0.0)

                if item_to_category and recs:
                    cats = {item_to_category.get(itm) for itm in recs if itm in item_to_category}
                    intra_list_diversities.append(len(cats) / len(recs))

                if item_popularity_scores and recs:
                    scores = [item_popularity_scores.get(itm, 0.0) for itm in recs]
                    pop_scores.append(float(np.mean(scores)))

            results[f"Precision@{k}"] = round(float(np.mean(precisions)), 4)
            results[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)
            results[f"HitRate@{k}"] = round(float(np.mean(hit_rates)), 4)
            results[f"NDCG@{k}"] = round(float(np.mean(ndcgs)), 4)
            results[f"MRR@{k}"] = round(float(np.mean(mrrs)), 4)

            coverage = len(all_recommended_items_at_k) / len(all_catalog_items) if all_catalog_items else 0.0
            results[f"CatalogCoverage@{k}"] = round(float(coverage * 100), 2)

            if intra_list_diversities:
                results[f"CategoryDiversity@{k}"] = round(float(np.mean(intra_list_diversities)), 4)

            if pop_scores:
                results[f"AvgPopularityBias@{k}"] = round(float(np.mean(pop_scores)), 4)

        return results

    @staticmethod
    def build_ground_truth(test_df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Vectorized construction of ground truth dictionary {visitorid: set(itemid)}."""
        grouped = test_df.groupby("visitorid")["itemid"].agg(set)
        return {int(uid): val for uid, val in grouped.items()}

