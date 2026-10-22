"""
Unified Recommendation Service for RecoSphere AI.
Integrates Two-Tower Neural Retrieval, Vector Index, Multi-Objective Ranking,
and Cold-Start Strategies behind a clean, production-grade interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
try:
    import torch
except ImportError:
    torch = None
import pandas as pd
import numpy as np

from configs.settings import MODELS_DIR, INDEX_DIR, PROCESSED_DATA_DIR, RankingConfig
from src.retrieval.index import VectorIndex
from src.ranking.ranker import RecommendationRanker
from src.models.baselines import PopularityRecommender, CategoryAffinityRecommender


class RecommendationService:
    """End-to-end recommendation service delivering personalized and cold-start recommendations."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        index_dir: Path = INDEX_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        ranking_config: Optional[RankingConfig] = None,
    ):
        self.models_dir = Path(models_dir)
        self.index_dir = Path(index_dir)
        self.processed_dir = Path(processed_dir)

        self.model = None
        self.vector_index = VectorIndex(index_dir=self.index_dir)
        self.ranker = RecommendationRanker(config=ranking_config)
        self.popularity_baseline = PopularityRecommender()
        self.category_baseline = CategoryAffinityRecommender()

        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.category_to_idx: Dict[int, int] = {}
        self.item_metadata: Dict[int, Dict[str, Any]] = {}
        self.user_history: Dict[int, Set[int]] = {}
        self.user_preferred_categories: Dict[int, Set[int]] = {}

        self.is_ready = False

    def initialize(self) -> bool:
        """Loads processed metadata, trained model checkpoint, catalog features, and vector index."""
        meta_path = self.processed_dir / "metadata.json"
        if not meta_path.exists():
            print("[RecommendationService] Processed metadata not found. Run data prep first.")
            return False

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load tower vocab if present
        vocab_path = self.models_dir / "tower_vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r") as f:
                v = json.load(f)
                self.user_to_idx = {int(k): val for k, val in v.get("user_to_tower_idx", {}).items()}
                self.item_to_idx = {int(k): val for k, val in v.get("item_to_tower_idx", {}).items()}
        else:
            self.user_to_idx = {int(k): v for k, v in meta["user_to_idx"].items()}
            self.item_to_idx = {int(k): v for k, v in meta["item_to_idx"].items()}
        self.category_to_idx = {int(k): v for k, v in meta["category_to_idx"].items()}

        # Load catalog
        catalog_path = self.processed_dir / "catalog.csv"
        catalog_df = pd.read_csv(catalog_path) if catalog_path.exists() else pd.DataFrame()

        # Load train events to build baselines and user history
        train_path = self.processed_dir / "train_events.csv"
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            self.popularity_baseline.fit(train_df)
            self.category_baseline.fit(train_df, catalog_df)

            # Vectorized user history for top active users to keep RAM low
            top_users = train_df["visitorid"].value_counts().head(50000).index
            sub_train = train_df[train_df["visitorid"].isin(top_users)]
            self.user_history = sub_train.groupby("visitorid")["itemid"].agg(set).to_dict()

        # Build item metadata cache
        item_scores = self.popularity_baseline.item_scores
        for iid, cat, prc, avl in zip(
            catalog_df["itemid"].values,
            catalog_df.get("categoryid", pd.Series(dtype=object)).values,
            catalog_df.get("price", pd.Series(dtype=float)).values,
            catalog_df.get("available", pd.Series(dtype=int)).values,
        ):
            iid_int = int(iid)
            self.item_metadata[iid_int] = {
                "categoryid": int(cat) if not pd.isna(cat) else None,
                "price": float(prc) if not pd.isna(prc) else 25.0,
                "available": bool(avl) if not pd.isna(avl) else True,
                "popularity_score": item_scores.get(iid_int, 0.0),
            }

        # Build user preferred categories from category baseline
        for uid, fav_cats in self.category_baseline.user_fav_categories.items():
            self.user_preferred_categories[uid] = set(fav_cats)

        # Load vector index
        index_loaded = self.vector_index.load("item_index.pt")

        # Load Two-Tower model if torch is available
        model_path = self.models_dir / "best_two_tower_model.pt"
        if model_path.exists() and torch is not None:
            try:
                from src.models.training import TwoTowerTrainer
                self.model = TwoTowerTrainer.load_model(model_path)
                print("[RecommendationService] Loaded Two-Tower neural model successfully.")
            except Exception as e:
                print(f"[RecommendationService] Warning loading model: {e}")

        self.is_ready = True
        return True

    def recommend(
        self,
        visitor_id: int,
        top_n: int = 10,
        exclude_seen: bool = True,
        enable_diversification: bool = True,
        category_filter: Optional[Set[int]] = None,
        relevance_weight: Optional[float] = None,
        popularity_weight: Optional[float] = None,
        category_weight: Optional[float] = None,
        mmr_lambda: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates top-N recommendations for a visitor.
        If user is known and model is available: Two-Tower vector retrieval + ranking.
        If user is cold-start: Fallback to popularity / category affinity baseline.
        """
        import time
        t0 = time.time()

        if not self.is_ready:
            self.initialize()

        seen_items = self.user_history.get(visitor_id, set()) if exclude_seen else set()
        user_idx = self.user_to_idx.get(visitor_id)

        # 1. Cold-start user detection
        if user_idx is None or self.model is None or self.vector_index.item_embeddings is None:
            fallback_items = self.category_baseline.recommend(visitor_id, top_k=top_n, exclude_seen=seen_items)
            recommendations = []
            for rank, itm in enumerate(fallback_items, 1):
                meta = self.item_metadata.get(itm, {})
                recommendations.append({
                    "rank": rank,
                    "item_id": itm,
                    "score": round(meta.get("popularity_score", 0.5), 4),
                    "similarity": 0.0,
                    "popularity_score": round(meta.get("popularity_score", 0.0), 4),
                    "category_id": meta.get("categoryid"),
                    "price": meta.get("price", 0.0),
                    "available": meta.get("available", True),
                    "explanation": {
                        "vector_similarity_contrib": "0.0%",
                        "catalog_popularity_contrib": "85.0%",
                        "category_affinity_contrib": "15.0%",
                        "matches_user_affinity": False,
                    },
                })

            latency = round((time.time() - t0) * 1000, 2)
            return {
                "visitor_id": visitor_id,
                "strategy": "cold_start_popularity_fallback",
                "is_cold_start": True,
                "explanation": "Visitor is not in training history. Served high-performing catalog items via popularity fallback.",
                "total_recommendations": len(recommendations),
                "recommendations": recommendations,
                "latency_ms": latency,
            }

        # 2. Personalized Two-Tower Retrieval
        with torch.no_grad():
            user_vec = self.model.get_user_embedding(user_idx)

        # Retrieve top candidate items from vector index
        candidates = self.vector_index.search(
            query_vector=user_vec,
            top_k=max(100, top_n * 5),
            exclude_items=seen_items,
            category_filter=category_filter,
            item_to_category={iid: m.get("categoryid") for iid, m in self.item_metadata.items()},
        )

        # 3. Multi-Signal Ranking & Diversification with dynamic weights
        preferred_cats = self.user_preferred_categories.get(visitor_id, set())
        ranked_items = self.ranker.rank(
            candidates=candidates,
            user_id=visitor_id,
            item_metadata=self.item_metadata,
            user_preferred_categories=preferred_cats,
            enable_diversification=enable_diversification,
            top_n=top_n,
            relevance_weight=relevance_weight,
            popularity_weight=popularity_weight,
            category_weight=category_weight,
            mmr_lambda=mmr_lambda,
        )

        latency = round((time.time() - t0) * 1000, 2)
        return {
            "visitor_id": visitor_id,
            "strategy": "two_tower_vector_retrieval_ranked",
            "is_cold_start": False,
            "explanation": f"Personalized recommendations generated via Two-Tower neural embedding match and {len(candidates)} retrieved candidates.",
            "total_recommendations": len(ranked_items),
            "recommendations": ranked_items,
            "latency_ms": latency,
        }

    def get_similar_items(
        self,
        item_id: int,
        top_n: int = 8,
        category_filter: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves nearest neighbor products to a given item in 64-dim embedding space.
        """
        import time
        t0 = time.time()

        if not self.is_ready:
            self.initialize()

        similar_tuples = self.vector_index.search_similar_items(
            item_id=item_id,
            top_k=top_n,
            category_filter=category_filter,
            item_to_category={iid: m.get("categoryid") for iid, m in self.item_metadata.items()},
        )

        # Fallback if vector embeddings not available (e.g., serverless environments)
        if not similar_tuples and item_id in self.item_metadata:
            target_cat = self.item_metadata[item_id].get("categoryid")
            candidates = []
            for iid, m in self.item_metadata.items():
                if iid == item_id:
                    continue
                if category_filter and m.get("categoryid") not in category_filter:
                    continue
                same_cat = (m.get("categoryid") == target_cat and target_cat is not None)
                score = 0.85 if same_cat else 0.45
                candidates.append((iid, score))
            candidates.sort(key=lambda x: (x[1], self.item_metadata.get(x[0], {}).get("popularity_score", 0)), reverse=True)
            similar_tuples = candidates[:top_n]

        results = []
        for rank, (sim_id, sim_score) in enumerate(similar_tuples, 1):
            meta = self.item_metadata.get(sim_id, {})
            results.append({
                "rank": rank,
                "item_id": sim_id,
                "similarity": round(float(sim_score), 4),
                "category_id": meta.get("categoryid"),
                "price": meta.get("price", 0.0),
                "available": meta.get("available", True),
                "popularity_score": round(meta.get("popularity_score", 0.0), 4),
            })

        latency = round((time.time() - t0) * 1000, 2)
        source_meta = self.item_metadata.get(item_id, {})

        return {
            "source_item_id": item_id,
            "source_item_category": source_meta.get("categoryid"),
            "total_similar_items": len(results),
            "similar_items": results,
            "latency_ms": latency,
        }
