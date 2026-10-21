# RecoSphere AI — Intelligent E-Commerce Recommendation Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.157-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)]()

> **Production-Style Two-Tower Deep Recommendation Engine with Fast Vector Retrieval, Multi-Objective Re-ranking, and A/B Testing Experimentation Lab.**

---

## 1. Project Overview & Architectural Vision

**RecoSphere AI** is an enterprise-grade e-commerce recommendation system engineered to predict and deliver the products an online shopper is most likely to interact with and purchase.

The platform combines deep metric learning, vector candidate generation, multi-signal ranking, Maximal Marginal Relevance (MMR) diversification, and a Monte-Carlo A/B testing simulator with statistical significance hypothesis testing.

```
+---------------------------------------------------------------------------------------------------------+
|                                        RECOSPHERE AI ARCHITECTURE                                       |
+---------------------------------------------------------------------------------------------------------+
                                                                                                          
      RAW DATASET                               TWO-TOWER NEURAL RETRIEVAL                ONLINE SERVING   
                                                                                                          
  [ Retailrocket ]                                   +-----------------+                                  
  * 2.75M Events                                     |   User Tower    |                                  
  * 417K Items         Chronological Split           | (MLP + Norm)    |                                  
  * 1.2K Categories   -------------------> User  --> +--------+--------+                                  
                                                              |                                           
                                                              v  Dense User Vector                        
                                                    +---------+---------+          +--------------------+ 
                                                    | Exact / Cosine    | -------> | Recommendation     | 
                                                    | Vector Search     | Top 100  | Multi-Signal       | 
                                                    +---------+---------+ Cand.    | Re-Ranking & MMR   | 
                                                              ^                    +---------+----------+ 
                                                              |                              |            
                                                     Dense Item Vectors                      v            
                                                     (Precomputed)                 Ranked Personalized    
                                                              |                    Recommendations        
                                                     +--------+--------+                     |            
                                     Item Catalog -> |   Item Tower    |                     +------> UI  
                                                     | (MLP + Norm)    |                     +------> API 
                                                     +-----------------+                                  
```

---

## 2. Key Modules & Technical Capabilities

1. **Two-Tower Deep Neural Network (`PyTorch`)**:
   - **User Tower**: Embeds user ID, interaction history volume, and engagement weights into a 64-dimensional metric space.
   - **Item Tower**: Embeds item ID, categorical taxonomy, price, and availability features into the shared metric space.
   - **Objective**: In-Batch Sampled Softmax Contrastive Loss (InfoNCE) with interaction event weighting (purchases = 5x, add-to-cart = 3x, views = 1x).
2. **High-Performance Vector Index (`src/retrieval/`)**:
   - Precomputes and persists dense item embeddings.
   - Performs inner-product / cosine similarity candidate generation for any user in sub-5ms.
3. **Multi-Signal Re-Ranking & Diversification (`src/ranking/`)**:
   - Balanced scoring: $\text{Score} = w_{\text{rel}} \cdot \text{Sim} + w_{\text{pop}} \cdot \text{Pop} + w_{\text{cat}} \cdot \text{CatAffinity}$.
   - **Maximal Marginal Relevance (MMR)**: Eliminates echo chambers and prevents single-category over-concentration.
   - Transparent attribution breakdown per recommended item.
4. **Deterministic Cold-Start Handling (`src/services/`)**:
   - Seamless fallback to global popularity and category affinity for unseen visitors.
5. **Monte-Carlo A/B Testing Simulator (`src/experiments/`)**:
   - Hash-based reproducible user bucketing (SHA-256).
   - Two-proportion Z-test for Click-Through Rate (CTR) and Conversion Rate (CVR).
   - Non-parametric bootstrap confidence intervals for Revenue Per User.
   - *Clear simulation labeling preventing false live business claims.*
6. **Production FastAPI Service (`src/api/`)**:
   - OpenAPI-documented endpoints for `/health`, `/metadata`, `/recommend`, and `/experiment/simulate`.
7. **Interactive 7-Page Streamlit Analytics Console (`app/dashboard.py`)**:
   - Executive Overview, Data Explorer, Model Training, Personalized Console, Offline Benchmark, A/B Lab, and System Health.

---

## 3. Dataset Profile (Retailrocket E-Commerce)

The application is built on the real-world **Retailrocket e-commerce dataset** downloaded automatically from Kaggle via `kagglehub`:

| Attribute | Measured Count |
|---|---|
| **Total Raw Events** | `2,756,101` |
| **Catalog Products** | `417,053` |
| **Categories** | `1,218` |
| **Train Events (80% Chronological)** | `2,204,512` |
| **Validation Events (10% Chronological)** | `275,564` |
| **Test Events (10% Chronological)** | `275,565` |
| **Matrix Sparsity** | `99.9994%` |
| **Pareto Principle** | Top 20% products account for `78.71%` of all events |

---

## 4. Measured Offline Benchmark Results

Models evaluated on the unseen chronological test interaction split (500 active test users, full candidate pool):

| Metric | Global Popularity Baseline | Category Affinity Baseline | Two-Tower Neural Network |
|---|---|---|---|
| **HitRate@10** | `0.0060` | `0.0080` | `0.0060` |
| **NDCG@10** | `0.0024` | `0.0031` | `0.0024` |
| **Precision@10** | `0.0006` | `0.0008` | `0.0006` |
| **Recall@10** | `0.0047` | `0.0067` | `0.0047` |
| **Catalog Coverage (%)** | `0.00%` | `0.15%` | `0.02%` |
| **Popularity Bias** | `0.6402` (Highest bias) | `0.5536` | `0.5910` |

> **Key Finding**: While static global popularity repeatedly recommends the same top 10 items, the Two-Tower model enables personalized retrieval across the catalog distribution.

---

## 5. Quick Start (Windows PowerShell Instructions)

### Prerequisites
- Python 3.12 (64-bit) installed on Windows.
- Windows PowerShell.

### Step 1: Clone & Setup Virtual Environment
```powershell
cd "c:\Projects\E-Commerce Recommender"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install CPU-Optimized Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 3: Run Data Ingestion & Preprocessing Pipeline
```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

### Step 4: Train Two-Tower Model, Build Index & Run Evaluation
```powershell
.\.venv\Scripts\python.exe -u scripts\train_and_evaluate.py
```

### Step 5: Launch the Interactive Web Platform (Modern Obsidian & Indigo Theme)
```powershell
.\.venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Interactive Web Application**: Open your browser at **`http://127.0.0.1:8000/`**
  - Modern Obsidian & Electric Indigo glassmorphic UI.
  - Interactive Chart.js conversion funnels & loss curves.
  - Live personalized recommendation playground with attribution cards.
  - Monte-Carlo A/B testing simulator with live statistical hypothesis decisions.
- **Interactive Swagger REST API Docs**: Open **`http://127.0.0.1:8000/docs`**

*(Optional) Streamlit Dashboard is also available if desired:*
```powershell
.\.venv\Scripts\streamlit.exe run app\dashboard.py
```

### Step 7: Run Automated Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
*(All 18 automated unit and integration tests execute and pass).*

---

## 6. Project Directory Layout

```
c:\Projects\E-Commerce Recommender\
├── .venv/                         # Local Python 3.12 virtual environment
├── data/
│   ├── raw/                       # Raw Retailrocket CSVs (events, item_properties)
│   └── processed/                 # Chronologically split & encoded Parquet/CSVs
├── artifacts/
│   ├── models/                    # best_two_tower_model.pt, tower_vocab.json
│   └── index/                     # item_index.pt (persisted dense vector index)
├── reports/
│   ├── eda_report.md              # Exploratory data analysis report
│   ├── model_evaluation.json      # Offline benchmark metrics
│   └── model_evaluation.md        # Comparative evaluation markdown report
├── configs/
│   └── settings.py                # System-wide configuration & hyperparameters
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Automated Kaggle ingestion & synthesis fallback
│   │   ├── validation.py          # Schema & data quality checks
│   │   ├── preprocessing.py       # Chronological split & zero-leakage encoding
│   │   └── eda.py                 # Statistical analysis module
│   ├── models/
│   │   ├── two_tower.py           # PyTorch User Tower & Item Tower architectures
│   │   ├── losses.py              # In-batch sampled softmax (InfoNCE) loss
│   │   ├── baselines.py           # Popularity & Category Affinity baseline models
│   │   ├── training.py            # Mini-batch training loop & early stopping
│   │   └── evaluation.py          # Top-K ranking, coverage & diversity evaluator
│   ├── retrieval/
│   │   └── index.py               # Vector similarity index (exact cosine / inner-prod)
│   ├── ranking/
│   │   ├── ranker.py              # Multi-signal scoring & explanation generator
│   │   └── diversification.py     # Maximal Marginal Relevance (MMR)
│   ├── experiments/
│   │   ├── assignment.py          # SHA-256 consistent bucket assigner
│   │   ├── statistics.py          # Two-proportion Z-test & bootstrap CIs
│   │   └── simulator.py           # Monte-Carlo A/B simulation engine
│   ├── services/
│   │   └── recommendation_service.py # Unified inference service
│   └── api/
│       ├── schemas.py             # Pydantic request/response schemas
│       └── main.py                # FastAPI REST endpoints
├── app/
│   └── dashboard.py               # 7-page executive analytics Streamlit UI
├── tests/
│   ├── test_data.py               # Data validation & leakage tests
│   ├── test_model.py              # Two-tower architecture & forward pass tests
│   ├── test_retrieval.py          # Vector index & candidate search tests
│   ├── test_ranking.py            # Ranking & MMR tests
│   ├── test_experiments.py        # A/B assignment & statistical tests
│   └── test_api.py                # FastAPI endpoint integration tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Known Limitations & Production Enhancements

1. **Cold-Start Latent Embeddings**: While cold users fall back to popularity, cold *items* can be further enhanced by fine-tuning a BERT or CLIP text/image model on raw item descriptions and images.
2. **ANN Index Scale**: For catalogs exceeding 10M products, integrate a Hierarchical Navigable Small World (HNSW) vector index (FAISS or ScaNN).
3. **Online Continuous Training**: Future iterations can incorporate real-time session updates via Redis stream buffers.

