# RecoSphere AI — Exploratory Data Analysis Report

## 1. Catalog & Interaction Overview
- **Total Interaction Events**: 2,756,101
- **Unique Users**: 1,407,580
- **Unique Products**: 235,061
- **Interaction Matrix Sparsity**: 99.9994%
- **Date Range**: 2015-05-03 03:00:04.384000 to 2015-09-18 02:59:47.788000 (138.0 days)

## 2. Event Type Distribution
| Event Type | Count | Share (%) |
|---|---|---|
| `view` | 2,664,312 | 96.67% |
| `addtocart` | 69,332 | 2.52% |
| `transaction` | 22,457 | 0.81% |

## 3. Power-Law & Long-Tail Dynamics
- **Pareto Principle**: Top 20% most popular items account for **78.71%** of all interaction volume.
- **User Activity Distribution**:
  - Median interactions per user: **1.0**
  - 75th percentile: **2.0**
  - Max interactions by a single user: **7757**
- **Item Popularity Distribution**:
  - Median events per item: **3.0**
  - Max events for a single item: **3412**

## 4. Chronological Split & Cold-Start Dynamics
- **Train Set**: 1,123,765 users, 212,915 items
- **Validation Set**: 156,954 users (90.29% cold start)
- **Test Set**: 158,020 users (92.67% cold start)
- **Unseen Cold Items (Test)**: 12947 items
