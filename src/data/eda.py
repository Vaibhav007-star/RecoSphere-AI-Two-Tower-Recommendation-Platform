"""
Exploratory Data Analysis (EDA) Module for RecoSphere AI.
Analyzes user-item interaction dynamics, catalog distributions,
matrix sparsity, and temporal coverage.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from configs.settings import REPORTS_DIR


class DataAnalyzer:
    """Computes production-grade EDA metrics and distributions for e-commerce interaction datasets."""

    def __init__(self, reports_dir: Path = REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        events_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Runs full suite of statistical exploratory analyses on the interaction dataset."""
        n_events = len(events_df)
        n_users = int(events_df["visitorid"].nunique())
        n_items = int(events_df["itemid"].nunique())

        # 1. Event type breakdown
        event_counts = events_df["event"].value_counts().to_dict()
        event_proportions = (events_df["event"].value_counts(normalize=True) * 100).round(2).to_dict()

        # 2. Sparsity of user-item matrix
        total_possible_cells = n_users * n_items
        unique_pairs = len(events_df[["visitorid", "itemid"]].drop_duplicates())
        sparsity = (1.0 - (unique_pairs / total_possible_cells)) * 100 if total_possible_cells > 0 else 0.0

        # 3. User activity distribution
        user_counts = events_df["visitorid"].value_counts()
        user_stats = {
            "mean": float(user_counts.mean()),
            "median": float(user_counts.median()),
            "std": float(user_counts.std()),
            "min": int(user_counts.min()),
            "p25": float(user_counts.quantile(0.25)),
            "p75": float(user_counts.quantile(0.75)),
            "p95": float(user_counts.quantile(0.95)),
            "max": int(user_counts.max()),
        }

        # 4. Item popularity distribution (Zipfian analysis)
        item_counts = events_df["itemid"].value_counts()
        item_stats = {
            "mean": float(item_counts.mean()),
            "median": float(item_counts.median()),
            "std": float(item_counts.std()),
            "min": int(item_counts.min()),
            "p25": float(item_counts.quantile(0.25)),
            "p75": float(item_counts.quantile(0.75)),
            "p95": float(item_counts.quantile(0.95)),
            "max": int(item_counts.max()),
        }

        # 5. Long-tail catalog analysis (Pareto: 20% most popular items share of total events)
        top_20_pct_count = max(1, int(n_items * 0.20))
        top_20_pct_events = item_counts.head(top_20_pct_count).sum()
        pareto_share = float((top_20_pct_events / n_events) * 100)

        # 6. Top popular products & categories
        top_10_products = item_counts.head(10).to_dict()

        category_analysis = {}
        if "categoryid" in catalog_df.columns:
            cat_counts = catalog_df["categoryid"].value_counts().head(10).to_dict()
            category_analysis["top_categories_catalog"] = {str(k): int(v) for k, v in cat_counts.items()}

        # 7. Temporal range
        ts_min = pd.to_datetime(events_df["timestamp"].min(), unit="ms")
        ts_max = pd.to_datetime(events_df["timestamp"].max(), unit="ms")
        duration_days = (ts_max - ts_min).total_seconds() / 86400.0

        # 8. Train/Val/Test coverage and cold-start stats
        train_users = set(train_df["visitorid"].unique())
        train_items = set(train_df["itemid"].unique())

        val_users = set(val_df["visitorid"].unique())
        val_items = set(val_df["itemid"].unique())
        test_users = set(test_df["visitorid"].unique())
        test_items = set(test_df["itemid"].unique())

        val_cold_users = val_users - train_users
        test_cold_users = test_users - train_users
        val_cold_items = val_items - train_items
        test_cold_items = test_items - train_items

        coverage_stats = {
            "train_users": len(train_users),
            "train_items": len(train_items),
            "val_users": len(val_users),
            "val_items": len(val_items),
            "test_users": len(test_users),
            "test_items": len(test_items),
            "val_cold_users_count": len(val_cold_users),
            "val_cold_users_pct": round(len(val_cold_users) / len(val_users) * 100, 2) if val_users else 0,
            "test_cold_users_count": len(test_cold_users),
            "test_cold_users_pct": round(len(test_cold_users) / len(test_users) * 100, 2) if test_users else 0,
            "val_cold_items_count": len(val_cold_items),
            "test_cold_items_count": len(test_cold_items),
        }

        report = {
            "total_events": n_events,
            "total_users": n_users,
            "total_items": n_items,
            "event_counts": event_counts,
            "event_proportions_pct": event_proportions,
            "matrix_sparsity_pct": round(sparsity, 4),
            "user_activity": user_stats,
            "item_popularity": item_stats,
            "pareto_top20_item_event_share_pct": round(pareto_share, 2),
            "top_10_products": {str(k): int(v) for k, v in top_10_products.items()},
            "category_analysis": category_analysis,
            "temporal": {
                "start_time": str(ts_min),
                "end_time": str(ts_max),
                "duration_days": round(duration_days, 1),
            },
            "split_coverage": coverage_stats,
        }

        # Write markdown report
        self._write_markdown_report(report)
        return report

    def _write_markdown_report(self, report: Dict[str, Any]):
        md_path = self.reports_dir / "eda_report.md"
        content = f"""# RecoSphere AI — Exploratory Data Analysis Report

## 1. Catalog & Interaction Overview
- **Total Interaction Events**: {report['total_events']:,}
- **Unique Users**: {report['total_users']:,}
- **Unique Products**: {report['total_items']:,}
- **Interaction Matrix Sparsity**: {report['matrix_sparsity_pct']}%
- **Date Range**: {report['temporal']['start_time']} to {report['temporal']['end_time']} ({report['temporal']['duration_days']} days)

## 2. Event Type Distribution
| Event Type | Count | Share (%) |
|---|---|---|
"""
        for event, count in report["event_counts"].items():
            pct = report["event_proportions_pct"].get(event, 0)
            content += f"| `{event}` | {count:,} | {pct}% |\n"

        content += f"""
## 3. Power-Law & Long-Tail Dynamics
- **Pareto Principle**: Top 20% most popular items account for **{report['pareto_top20_item_event_share_pct']}%** of all interaction volume.
- **User Activity Distribution**:
  - Median interactions per user: **{report['user_activity']['median']:.1f}**
  - 75th percentile: **{report['user_activity']['p75']:.1f}**
  - Max interactions by a single user: **{report['user_activity']['max']}**
- **Item Popularity Distribution**:
  - Median events per item: **{report['item_popularity']['median']:.1f}**
  - Max events for a single item: **{report['item_popularity']['max']}**

## 4. Chronological Split & Cold-Start Dynamics
- **Train Set**: {report['split_coverage']['train_users']:,} users, {report['split_coverage']['train_items']:,} items
- **Validation Set**: {report['split_coverage']['val_users']:,} users ({report['split_coverage']['val_cold_users_pct']}% cold start)
- **Test Set**: {report['split_coverage']['test_users']:,} users ({report['split_coverage']['test_cold_users_pct']}% cold start)
- **Unseen Cold Items (Test)**: {report['split_coverage']['test_cold_items_count']} items
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[DataAnalyzer] EDA report generated at {md_path}")

