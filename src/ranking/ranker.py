"""
Recommendation Ranker and Explainer Module for RecoSphere AI.
Combines vector retrieval similarity with popularity, category affinities,
and diversity constraints (MMR) while providing signal-level transparency.
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
import pandas as pd

from configs.settings import RankingConfig
from src.ranking.diversification import MMRDiversifier


class RecommendationRanker:
    """
    Ranks candidate items using multi-objective scoring:
    Score = w_rel * Similarity + w_pop * Popularity + w_cat * CategoryAffinity
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.diversifier = MMRDiversifier(lmbda=self.config.mmr_lambda)

    def rank(
        self,
        candidates: List[tuple],  # List of (item_id, similarity_score)
        user_id: int,
        item_metadata: Dict[int, Dict[str, Any]],
        user_preferred_categories: Optional[Set[int]] = None,
        enable_diversification: bool = True,
        top_n: Optional[int] = None,
        relevance_weight: Optional[float] = None,
        popularity_weight: Optional[float] = None,
        category_weight: Optional[float] = None,
        mmr_lambda: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate products, applies optional MMR diversification,
        and outputs structured explanations for each recommended item.
        """
        top_n = top_n or self.config.final_top_n
        if not candidates:
            return []

        user_cats = user_preferred_categories or set()
        w_rel = relevance_weight if relevance_weight is not None else self.config.relevance_weight
        w_pop = popularity_weight if popularity_weight is not None else self.config.popularity_weight
        w_cat = category_weight if category_weight is not None else self.config.diversity_weight
        current_lmbda = mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda

        scored_candidates = []
        item_to_cat = {}

        # Max raw similarity for normalization
        raw_sims = [sim for _, sim in candidates]
        min_sim, max_sim = min(raw_sims), max(raw_sims)
        sim_range = max_sim - min_sim if max_sim > min_sim else 1.0

        for item_id, sim in candidates:
            meta = item_metadata.get(item_id, {})
            cat_id = meta.get("categoryid")
            if cat_id is not None:
                item_to_cat[item_id] = cat_id

            # Normalized vector relevance [0, 1]
            norm_sim = (sim - min_sim) / sim_range

            # Popularity score [0, 1]
            pop_score = meta.get("popularity_score", 0.0)

            # Category affinity boost [0 or 1]
            cat_match = 1.0 if cat_id in user_cats else 0.0

            # Composite ranking score
            composite_score = (w_rel * norm_sim) + (w_pop * pop_score) + (w_cat * cat_match)

            # Signal attribution breakdown
            total_w = w_rel + w_pop + w_cat
            rel_pct = round((w_rel * norm_sim) / total_w * 100, 1) if total_w > 0 else 0
            pop_pct = round((w_pop * pop_score) / total_w * 100, 1) if total_w > 0 else 0
            cat_pct = round((w_cat * cat_match) / total_w * 100, 1) if total_w > 0 else 0

            scored_candidates.append({
                "item_id": int(item_id),
                "score": float(composite_score),
                "similarity": float(sim),
                "popularity_score": float(pop_score),
                "category_id": cat_id,
                "price": meta.get("price", 0.0),
                "available": bool(meta.get("available", True)),
                "explanation": {
                    "vector_similarity_contrib": f"{rel_pct}%",
                    "catalog_popularity_contrib": f"{pop_pct}%",
                    "category_affinity_contrib": f"{cat_pct}%",
                    "matches_user_affinity": cat_match > 0,
                },
            })

        # Sort by composite score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Diversification
        if enable_diversification:
            self.diversifier.lmbda = current_lmbda
            tuple_candidates = [(x["item_id"], x["score"]) for x in scored_candidates]
            diversified_tuples = self.diversifier.diversify(tuple_candidates, item_to_cat, top_n=top_n)
            diversified_ids = [t[0] for t in diversified_tuples]

            # Reorder scored_candidates based on diversified selection
            id_to_item = {x["item_id"]: x for x in scored_candidates}
            final_recs = [id_to_item[iid] for iid in diversified_ids if iid in id_to_item]
        else:
            final_recs = scored_candidates[:top_n]

        # Assign ranks
        for idx, rec in enumerate(final_recs):
            rec["rank"] = idx + 1

        return final_recs

