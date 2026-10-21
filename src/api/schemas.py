"""
Pydantic Request and Response Schemas for RecoSphere AI FastAPI Backend.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    visitor_id: int = Field(..., description="Unique Visitor / Customer ID", json_schema_extra={"example": 10245})
    top_n: int = Field(10, ge=1, le=100, description="Number of items to recommend")
    exclude_seen: bool = Field(True, description="Exclude items already interacted with in training history")
    enable_diversification: bool = Field(True, description="Apply MMR diversity constraint")
    category_filter: Optional[List[int]] = Field(None, description="Optional list of category IDs to restrict to")
    relevance_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for vector similarity")
    popularity_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for catalog popularity")
    category_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for category affinity")
    mmr_lambda: Optional[float] = Field(None, ge=0.0, le=1.0, description="MMR trade-off (1=relevance, 0=diversity)")


class BatchRecommendationRequest(BaseModel):
    visitor_ids: List[int] = Field(..., min_length=1, max_length=50, description="List of user IDs")
    top_n: int = Field(10, ge=1, le=50)
    exclude_seen: bool = Field(True)


class RecommendedItemSchema(BaseModel):
    rank: int
    item_id: int
    score: float
    similarity: float
    popularity_score: float
    category_id: Optional[int] = None
    price: float = 0.0
    available: bool = True
    explanation: Dict[str, Any]


class RecommendationResponse(BaseModel):
    visitor_id: int
    strategy: str
    is_cold_start: bool
    explanation: str
    total_recommendations: int
    recommendations: List[RecommendedItemSchema]
    latency_ms: Optional[float] = None


class SimilarItemRequest(BaseModel):
    item_id: int = Field(..., description="Source Item ID to find nearest neighbors for", json_schema_extra={"example": 460429})
    top_n: int = Field(8, ge=1, le=50)
    category_filter: Optional[List[int]] = None


class SimilarItemSchema(BaseModel):
    rank: int
    item_id: int
    similarity: float
    category_id: Optional[int] = None
    price: float = 0.0
    available: bool = True
    popularity_score: float = 0.0


class SimilarItemResponse(BaseModel):
    source_item_id: int
    source_item_category: Optional[int] = None
    total_similar_items: int
    similar_items: List[SimilarItemSchema]
    latency_ms: Optional[float] = None


class ExperimentSimulationRequest(BaseModel):
    sample_size_per_variant: int = Field(5000, ge=100, le=100000)
    baseline_ctr: float = Field(0.08, ge=0.001, le=1.0)
    baseline_cvr: float = Field(0.03, ge=0.001, le=1.0)
    baseline_addtocart: float = Field(0.06, ge=0.001, le=1.0)
    treatment_lift_ctr: float = Field(0.15, ge=-1.0, le=5.0)
    treatment_lift_cvr: float = Field(0.15, ge=-1.0, le=5.0)
    treatment_lift_addtocart: float = Field(0.15, ge=-1.0, le=5.0)
    avg_order_value_base: float = Field(45.0, ge=1.0)
    alpha: float = Field(0.05, ge=0.001, le=0.50)
    random_seed: int = Field(42)


class SampleSizeRequest(BaseModel):
    baseline_rate: float = Field(0.03, ge=0.001, le=0.99, description="Baseline conversion rate")
    mde_relative: float = Field(0.10, ge=0.01, le=1.0, description="Minimum detectable relative lift (e.g., 0.10 for +10%)")
    alpha: float = Field(0.05, ge=0.001, le=0.20)
    power: float = Field(0.80, ge=0.50, le=0.99)


class SampleSizeResponse(BaseModel):
    baseline_rate: float
    mde_relative: float
    target_rate: float
    required_sample_size_per_variant: int
    total_sample_size: int
    alpha: float
    power: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    index_loaded: bool
    catalog_items_count: int
