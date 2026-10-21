"""
RecoSphere AI — Configuration and System Constants
Centralized settings for data processing, model architecture, training, retrieval, and experimentation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
INDEX_DIR = ARTIFACTS_DIR / "index"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass
class DataConfig:
    min_user_interactions: int = 3
    min_item_interactions: int = 3
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42
    event_weights: dict = field(
        default_factory=lambda: {
            "view": 1.0,
            "addtocart": 3.0,
            "transaction": 5.0,
        }
    )


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    temperature: float = 0.07
    l2_reg: float = 1e-5
    learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 10
    patience: int = 3
    random_seed: int = 42
    device: str = "cpu"


@dataclass
class RetrievalConfig:
    top_k_candidates: int = 100
    similarity_metric: str = "cosine"  # cosine or dot_product
    index_type: str = "exact"  # exact or faiss/scann


@dataclass
class RankingConfig:
    relevance_weight: float = 0.70
    popularity_weight: float = 0.15
    diversity_weight: float = 0.15
    mmr_lambda: float = 0.6  # Maximal Marginal Relevance trade-off
    final_top_n: int = 10


@dataclass
class ExperimentConfig:
    default_sample_size: int = 5000
    alpha: float = 0.05
    power: float = 0.80
    assumed_baseline_conversion: float = 0.035  # 3.5%
    assumed_treatment_lift: float = 0.15  # 15% relative lift
    bootstrap_iterations: int = 1000
    random_seed: int = 42

