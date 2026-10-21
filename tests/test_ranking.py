"""
Automated Unit Tests for Multi-Objective Ranking and MMR Diversification.
"""

import pytest
from src.ranking.ranker import RecommendationRanker
from src.ranking.diversification import MMRDiversifier
from configs.settings import RankingConfig


def test_mmr_diversifier():
    diversifier = MMRDiversifier(lmbda=0.5)
    candidates = [(1, 0.95), (2, 0.90), (3, 0.88), (4, 0.85)]
    # Items 1, 2, 3 have category 10, item 4 has category 20
    item_to_cat = {1: 10, 2: 10, 3: 10, 4: 20}

    diversified = diversifier.diversify(candidates, item_to_cat, top_n=3)
    diversified_ids = [t[0] for t in diversified]

    assert len(diversified) == 3
    # Top item is chosen first
    assert diversified_ids[0] == 1
    # Category 20 (item 4) should be boosted over item 3 due to MMR diversity penalty
    assert 4 in diversified_ids


def test_ranker_multi_signal_attribution():
    ranker = RecommendationRanker(RankingConfig(final_top_n=3))
    candidates = [(101, 0.9), (102, 0.5), (103, 0.3)]
    item_meta = {
        101: {"categoryid": 1, "popularity_score": 0.8, "price": 20.0},
        102: {"categoryid": 2, "popularity_score": 0.2, "price": 40.0},
        103: {"categoryid": 3, "popularity_score": 0.1, "price": 10.0},
    }

    ranked = ranker.rank(
        candidates=candidates,
        user_id=999,
        item_metadata=item_meta,
        user_preferred_categories={2},
        enable_diversification=False,
        top_n=3,
    )

    assert len(ranked) == 3
    assert ranked[0]["rank"] == 1
    # Check explanation structure
    exp = ranked[0]["explanation"]
    assert "vector_similarity_contrib" in exp
    assert "catalog_popularity_contrib" in exp
    assert "category_affinity_contrib" in exp

