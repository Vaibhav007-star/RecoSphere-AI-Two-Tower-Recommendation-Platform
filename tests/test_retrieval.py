"""
Automated Unit Tests for Vector Index, Candidate Retrieval, and Item-to-Item Similarity Search.
"""

import pytest
import torch
from src.retrieval.index import VectorIndex


def test_vector_index_search():
    index = VectorIndex()
    item_ids = [101, 102, 103, 104, 105]
    raw = torch.randn(5, 16)
    embeddings = raw / torch.norm(raw, p=2, dim=-1, keepdim=True)

    index.build_index(item_ids, embeddings)

    q = embeddings[2].clone()
    results = index.search(query_vector=q, top_k=3)

    assert len(results) == 3
    top_item, top_score = results[0]
    assert top_item == 103
    assert abs(top_score - 1.0) < 1e-4


def test_vector_index_exclusion():
    index = VectorIndex()
    item_ids = [1, 2, 3, 4]
    embeddings = torch.randn(4, 8)
    embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    index.build_index(item_ids, embeddings)

    results = index.search(query_vector=embeddings[0], top_k=4, exclude_items={1, 2})
    retrieved_ids = [r[0] for r in results]

    assert 1 not in retrieved_ids
    assert 2 not in retrieved_ids
    assert set(retrieved_ids).issubset({3, 4})


def test_search_similar_items():
    index = VectorIndex()
    item_ids = [501, 502, 503, 504]
    embeddings = torch.randn(4, 16)
    embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    index.build_index(item_ids, embeddings)

    # Search neighbors of item 501
    similar = index.search_similar_items(item_id=501, top_k=2)
    assert len(similar) == 2
    # Should not contain 501 itself (it was excluded)
    sim_ids = [s[0] for s in similar]
    assert 501 not in sim_ids
