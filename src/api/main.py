"""
FastAPI Production Application for RecoSphere AI Recommender Engine.
Exposes REST endpoints for real-time personalized recommendations,
item-to-item similarity search, evaluation benchmarks, and A/B test simulation.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from configs.settings import REPORTS_DIR, BASE_DIR
from src.api.schemas import (
    RecommendationRequest,
    BatchRecommendationRequest,
    RecommendationResponse,
    SimilarItemRequest,
    SimilarItemResponse,
    ExperimentSimulationRequest,
    SampleSizeRequest,
    SampleSizeResponse,
    HealthResponse,
)
from src.services.recommendation_service import RecommendationService
from src.experiments.simulator import ABExperimentSimulator
from src.experiments.statistics import ExperimentStatistics

# Shared Service Instances
recommendation_service = RecommendationService()
simulator = ABExperimentSimulator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for safe startup initialization and cleanup."""
    try:
        recommendation_service.initialize()
        print("[API] Recommendation Service initialized on startup.")
    except Exception as e:
        print(f"[API] Warning during startup initialization: {e}")
    yield


app = FastAPI(
    title="RecoSphere AI — Recommendation API & Web Platform",
    description="Production-style Two-Tower Recommendation Engine with Fast Vector Retrieval & A/B Testing Simulator.",
    version="1.1.0",
    lifespan=lifespan,
)

# Mount static files for modern web interface
static_dir = BASE_DIR / "app" / "static"
if not static_dir.exists():
    static_dir = BASE_DIR / "public" / "static"
if not static_dir.exists():
    static_dir = Path("app/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
def serve_web_interface():
    """Serves the interactive RecoSphere AI web application."""
    index_path = BASE_DIR / "app" / "templates" / "index.html"
    if not index_path.exists():
        index_path = Path("app/templates/index.html")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RecoSphere AI Web App</h1>"


@app.get("/health", response_model=HealthResponse, tags=["System Health"])
def health_check():
    """Returns runtime health status, model artifact availability, and index size."""
    return HealthResponse(
        status="healthy" if recommendation_service.is_ready else "degraded",
        service="RecoSphere AI Recommendation Engine",
        version="1.1.0",
        model_loaded=recommendation_service.model is not None,
        index_loaded=recommendation_service.vector_index.item_embeddings is not None,
        catalog_items_count=len(recommendation_service.item_metadata),
    )


@app.get("/metadata", tags=["Metadata & Catalog"])
def get_metadata():
    """Returns overview of known users, catalog items, and system dimensions."""
    return {
        "num_known_users": len(recommendation_service.user_to_idx),
        "num_catalog_items": len(recommendation_service.item_metadata),
        "num_categories": len(recommendation_service.category_to_idx),
        "embedding_dim": (
            recommendation_service.vector_index.item_embeddings.size(1)
            if recommendation_service.vector_index.item_embeddings is not None
            else 64
        ),
        "similarity_metric": "cosine",
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Personalized Recommendations"])
def get_recommendations(req: RecommendationRequest):
    """
    Delivers top-N ranked recommendations for a given customer/visitor.
    Supports dynamic multi-objective weights and MMR diversification.
    """
    cat_filter = set(req.category_filter) if req.category_filter else None
    result = recommendation_service.recommend(
        visitor_id=req.visitor_id,
        top_n=req.top_n,
        exclude_seen=req.exclude_seen,
        enable_diversification=req.enable_diversification,
        category_filter=cat_filter,
        relevance_weight=req.relevance_weight,
        popularity_weight=req.popularity_weight,
        category_weight=req.category_weight,
        mmr_lambda=req.mmr_lambda,
    )
    return result


@app.post("/recommend/similar", response_model=SimilarItemResponse, tags=["Item-to-Item Similarity"])
def get_similar_items(req: SimilarItemRequest):
    """
    Item-to-Item neural nearest neighbor retrieval in 64-dim embedding space.
    """
    cat_filter = set(req.category_filter) if req.category_filter else None
    return recommendation_service.get_similar_items(
        item_id=req.item_id,
        top_n=req.top_n,
        category_filter=cat_filter,
    )


@app.get("/recommend/similar/{item_id}", response_model=SimilarItemResponse, tags=["Item-to-Item Similarity"])
def get_similar_items_by_id(item_id: int, top_n: int = 8):
    """Quick GET endpoint for item-to-item nearest neighbors."""
    return recommendation_service.get_similar_items(
        item_id=item_id,
        top_n=top_n,
    )


@app.post("/recommend/batch", response_model=List[RecommendationResponse], tags=["Personalized Recommendations"])
def get_batch_recommendations(req: BatchRecommendationRequest):
    """Batch recommendation endpoint for high-throughput user campaigns."""
    results = []
    for uid in req.visitor_ids:
        res = recommendation_service.recommend(
            visitor_id=uid,
            top_n=req.top_n,
            exclude_seen=req.exclude_seen,
        )
        results.append(res)
    return results


@app.get("/model/metrics", tags=["Model Evaluation"])
def get_model_evaluation_metrics():
    """Returns measured comparative offline benchmark metrics."""
    eval_path = REPORTS_DIR / "model_evaluation.json"
    if not eval_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not found. Run model evaluation first.",
        )
    with open(eval_path, "r") as f:
        return json.load(f)


@app.post("/experiment/simulate", tags=["A/B Testing Simulator"])
def simulate_experiment(req: ExperimentSimulationRequest):
    """
    Runs a controlled, reproducible Monte-Carlo A/B simulation comparing
    Popularity Baseline (Control) vs Two-Tower Engine (Treatment).
    """
    return simulator.run_simulation(
        sample_size_per_variant=req.sample_size_per_variant,
        baseline_ctr=req.baseline_ctr,
        baseline_cvr=req.baseline_cvr,
        baseline_addtocart=req.baseline_addtocart,
        treatment_lift_ctr=req.treatment_lift_ctr,
        treatment_lift_cvr=req.treatment_lift_cvr,
        treatment_lift_addtocart=req.treatment_lift_addtocart,
        avg_order_value_base=req.avg_order_value_base,
        alpha=req.alpha,
        random_seed=req.random_seed,
    )


@app.post("/experiment/sample-size", response_model=SampleSizeResponse, tags=["A/B Testing Simulator"])
def calculate_sample_size(req: SampleSizeRequest):
    """
    Calculates statistical sample size and power requirements for A/B testing.
    """
    n_per_variant = ExperimentStatistics.calculate_sample_size(
        baseline_rate=req.baseline_rate,
        mde_relative=req.mde_relative,
        alpha=req.alpha,
        power=req.power,
    )
    return SampleSizeResponse(
        baseline_rate=req.baseline_rate,
        mde_relative=req.mde_relative,
        target_rate=round(req.baseline_rate * (1 + req.mde_relative), 4),
        required_sample_size_per_variant=n_per_variant,
        total_sample_size=n_per_variant * 2,
        alpha=req.alpha,
        power=req.power,
    )

@app.post("/experiment/sample-size", response_model=SampleSizeResponse, tags=["A/B Testing Simulator"])
def calculate_sample_size(req: SampleSizeRequest):
    """
    Calculates statistical sample size and power requirements for A/B testing.
    """
    n_per_variant = ExperimentStatistics.calculate_sample_size(
        baseline_rate=req.baseline_rate,
        mde_relative=req.mde_relative,
        alpha=req.alpha,
        power=req.power,
    )
    return SampleSizeResponse(
        baseline_rate=req.baseline_rate,
        mde_relative=req.mde_relative,
        target_rate=round(req.baseline_rate * (1 + req.mde_relative), 4),
        required_sample_size_per_variant=n_per_variant,
        total_sample_size=n_per_variant * 2,
        alpha=req.alpha,
        power=req.power,
    )

