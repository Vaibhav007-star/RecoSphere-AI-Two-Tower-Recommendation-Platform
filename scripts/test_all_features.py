"""
End-to-End System Verification and Feature Testing Script for RecoSphere AI.
Validates all modules: Data Pipeline, PyTorch Neural Model, Vector Retrieval,
Item-to-Item Similarity, Dynamic Multi-Objective Ranking, A/B Testing, and Web API.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app, recommendation_service


def run_full_system_verification():
    print("=" * 80)
    print("RECOSPHERE AI — END-TO-END FEATURE VERIFICATION & SMOKE TEST")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. System Health & Metadata
        print("\n[Test 1/8] Checking System Health & Metadata Endpoints...")
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        health_data = res_health.json()
        print(f" -> Status: {health_data['status']} | Model Loaded: {health_data['model_loaded']} | Catalog: {health_data['catalog_items_count']:,} items")

        res_meta = client.get("/metadata")
        assert res_meta.status_code == 200
        meta_data = res_meta.json()
        print(f" -> Users in Vocab: {meta_data['num_known_users']:,} | Categories: {meta_data['num_categories']} | Embedding Dim: {meta_data['embedding_dim']}")

        # 2. Web Application Delivery
        print("\n[Test 2/8] Testing Web Interface Delivery...")
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "RecoSphere AI" in res_root.text
        print(f" -> Root Page (HTML): 200 OK ({len(res_root.text):,} bytes)")

        res_css = client.get("/static/css/styles.css")
        assert res_css.status_code == 200
        print(f" -> Stylesheet (CSS): 200 OK ({len(res_css.text):,} bytes)")

        res_js = client.get("/static/js/app.js")
        assert res_js.status_code == 200
        print(f" -> Frontend App (JS): 200 OK ({len(res_js.text):,} bytes)")

        # 3. Personalized Two-Tower Recommendations
        print("\n[Test 3/8] Testing Personalized Two-Tower Retrieval & Multi-Objective Ranking...")
        # Pick an active user that was in the Two-Tower vocabulary
        active_uid = list(recommendation_service.user_to_idx.keys())[0] if recommendation_service.user_to_idx else 10245
        payload_pers = {
            "visitor_id": active_uid,
            "top_n": 5,
            "exclude_seen": True,
            "enable_diversification": True,
            "relevance_weight": 0.70,
            "popularity_weight": 0.15,
            "category_weight": 0.15,
            "mmr_lambda": 0.60,
        }
        t0 = time.time()
        res_pers = client.post("/recommend", json=payload_pers)
        elapsed = (time.time() - t0) * 1000
        assert res_pers.status_code == 200
        data_pers = res_pers.json()
        print(f" -> User ID: {active_uid} | Strategy: {data_pers['strategy']} | Returned: {len(data_pers['recommendations'])} items | Latency: {data_pers['latency_ms']} ms (Total roundtrip: {elapsed:.2f} ms)")
        for r in data_pers['recommendations'][:3]:
            exp = r['explanation']
            print(f"    - Rank #{r['rank']} | Item {r['item_id']} | Score: {r['score']:.4f} | Sim: {r['similarity']:.4f} | Attribution: {exp['vector_similarity_contrib']} sim, {exp['catalog_popularity_contrib']} pop")

        # 4. Cold-Start Fallback Strategy
        print("\n[Test 4/8] Testing Cold-Start Handling for Unknown Visitor...")
        payload_cold = {"visitor_id": 888888888, "top_n": 5}
        res_cold = client.post("/recommend", json=payload_cold)
        assert res_cold.status_code == 200
        data_cold = res_cold.json()
        assert data_cold['is_cold_start'] is True
        print(f" -> Detected Cold-Start: {data_cold['is_cold_start']} | Fallback Strategy: {data_cold['strategy']}")
        print(f" -> Explanation: {data_cold['explanation']}")

        # 5. Item-to-Item Similar Products Vector Search
        print("\n[Test 5/8] Testing Item-to-Item Similarity Vector Retrieval...")
        test_item = recommendation_service.vector_index.item_ids[0] if recommendation_service.vector_index.item_ids else 102
        payload_sim = {"item_id": test_item, "top_n": 4}
        res_sim = client.post("/recommend/similar", json=payload_sim)
        assert res_sim.status_code == 200
        data_sim = res_sim.json()
        print(f" -> Source Product: {data_sim['source_item_id']} | Category: {data_sim['source_item_category']} | Retrieved: {data_sim['total_similar_items']} neighbors ({data_sim['latency_ms']} ms)")
        for sim in data_sim['similar_items']:
            print(f"    - Neighbor Item {sim['item_id']} | Cosine Similarity: {sim['similarity']:.4f} | Category: {sim['category_id']}")

        # 6. Batch Recommendations
        print("\n[Test 6/8] Testing Batch Recommendations Endpoint...")
        payload_batch = {"visitor_ids": [active_uid, 888888888], "top_n": 3}
        res_batch = client.post("/recommend/batch", json=payload_batch)
        assert res_batch.status_code == 200
        batch_data = res_batch.json()
        print(f" -> Processed batch of {len(batch_data)} visitors successfully.")

        # 7. A/B Testing Monte-Carlo Simulator
        print("\n[Test 7/8] Testing A/B Testing Simulator with Statistical Significance Tests...")
        payload_ab = {
            "sample_size_per_variant": 5000,
            "baseline_ctr": 0.08,
            "baseline_cvr": 0.03,
            "treatment_lift_ctr": 0.18,
            "treatment_lift_cvr": 0.15,
            "alpha": 0.05,
            "random_seed": 42,
        }
        res_ab = client.post("/experiment/simulate", json=payload_ab)
        assert res_ab.status_code == 200
        data_ab = res_ab.json()
        cvr_m = data_ab['metrics']['conversion_rate']
        rev_m = data_ab['metrics']['revenue_per_user']
        print(f" -> Simulation Flag: is_simulation={data_ab['is_simulation']}")
        print(f" -> Control CVR: {cvr_m['control_rate']*100:.2f}% | Treatment CVR: {cvr_m['treatment_rate']*100:.2f}% (Lift: +{cvr_m['relative_lift_pct']}%)")
        print(f" -> Z-Statistic: {cvr_m['z_statistic']:.3f} | p-value: {cvr_m['p_value']:.4f} | Significant: {cvr_m['is_statistically_significant']}")
        print(f" -> 95% Confidence Interval for Lift: [{cvr_m['ci_lower']*100:.2f}%, {cvr_m['ci_upper']*100:.2f}%]")
        print(f" -> Revenue per User Bootstrap 95% CI: [${rev_m['ci_lower']}, ${rev_m['ci_upper']}]")

        # 8. Statistical Power & Sample Size Calculator
        print("\n[Test 8/8] Testing Statistical Power & Sample Size Planning Calculator...")
        payload_pwr = {
            "baseline_rate": 0.035,
            "mde_relative": 0.10,
            "alpha": 0.05,
            "power": 0.80,
        }
        res_pwr = client.post("/experiment/sample-size", json=payload_pwr)
        assert res_pwr.status_code == 200
        data_pwr = res_pwr.json()
        print(f" -> Required Sample Size per Variant: {data_pwr['required_sample_size_per_variant']:,} users")
        print(f" -> Total Traffic for 80% Power: {data_pwr['total_sample_size']:,} visitors")

    print("\n" + "=" * 80)
    print("ALL 8 END-TO-END FEATURE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_system_verification()

