"""
Automated Unit Tests for FastAPI Application Endpoints.
Covers: Health, Metadata, Personalized Recs, Similar Items,
Dynamic Weights, Sample Size Power Analysis, and Root Web Interface.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_web_interface():
    response = client.get("/")
    assert response.status_code == 200
    assert "RecoSphere AI" in response.text
    assert "<!DOCTYPE html>" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "RecoSphere AI Recommendation Engine"


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "num_catalog_items" in data
    assert "similarity_metric" in data


def test_recommend_endpoint_cold_start():
    # Unknown visitor ID 999999999
    payload = {
        "visitor_id": 999999999,
        "top_n": 5,
        "exclude_seen": True,
        "enable_diversification": True,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visitor_id"] == 999999999
    assert data["is_cold_start"] is True
    assert len(data["recommendations"]) <= 5
    assert "explanation" in data
    assert "latency_ms" in data


def test_recommend_dynamic_weights_and_ranking():
    payload = {
        "visitor_id": 10245,
        "top_n": 6,
        "exclude_seen": False,
        "enable_diversification": True,
        "relevance_weight": 0.85,
        "popularity_weight": 0.10,
        "category_weight": 0.05,
        "mmr_lambda": 0.5,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) <= 6
    assert data["latency_ms"] is not None
    assert data["latency_ms"] >= 0.0


def test_similar_items_endpoints():
    # Test POST
    payload = {"item_id": 460429, "top_n": 5}
    response = client.post("/recommend/similar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["source_item_id"] == 460429
    assert "similar_items" in data
    assert "latency_ms" in data

    # Test GET
    get_res = client.get("/recommend/similar/460429?top_n=4")
    assert get_res.status_code == 200
    assert get_res.json()["source_item_id"] == 460429


def test_batch_recommendations():
    payload = {
        "visitor_ids": [10245, 9999999],
        "top_n": 4,
        "exclude_seen": True,
    }
    response = client.post("/recommend/batch", json=payload)
    assert response.status_code == 200
    batch_data = response.json()
    assert len(batch_data) == 2
    assert batch_data[0]["visitor_id"] == 10245
    assert batch_data[1]["visitor_id"] == 9999999


def test_simulate_experiment_endpoint():
    payload = {
        "sample_size_per_variant": 1000,
        "baseline_ctr": 0.05,
        "baseline_cvr": 0.02,
        "treatment_lift_cvr": 0.10,
    }
    response = client.post("/experiment/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_simulation"] is True
    assert data["sample_size_per_variant"] == 1000


def test_sample_size_power_calculator():
    payload = {
        "baseline_rate": 0.035,
        "mde_relative": 0.10,
        "alpha": 0.05,
        "power": 0.80,
    }
    response = client.post("/experiment/sample-size", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["required_sample_size_per_variant"] > 0
    assert data["total_sample_size"] == data["required_sample_size_per_variant"] * 2
