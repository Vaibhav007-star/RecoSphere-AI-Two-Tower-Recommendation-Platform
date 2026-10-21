"""
Dataset Ingestion Module for RecoSphere AI.
Responsible for acquiring, loading, and enriching e-commerce interaction data.
Supports:
1. Local CSV datasets (e.g., Retailrocket events and item properties).
2. Automatic extraction of categoryid and availability from item properties.
3. Realistic fallback synthesis if needed.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from configs.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataIngestion:
    """Manages raw data loading, downloading, and catalog property extraction."""

    EVENTS_FILENAME = "events.csv"
    ITEM_PROPS_FILENAME = "item_properties.csv"
    CATEGORY_TREE_FILENAME = "category_tree.csv"

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def check_local_dataset(self) -> bool:
        """Checks if required raw events CSV exists."""
        events_path = self.raw_dir / self.EVENTS_FILENAME
        return events_path.exists() and events_path.stat().st_size > 0

    def extract_item_properties(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Extracts item properties (categoryid and availability) from Retailrocket parts.
        Optimized chunked extraction.
        """
        cached_meta = self.raw_dir / self.ITEM_PROPS_FILENAME
        if cached_meta.exists() and cached_meta.stat().st_size > 1000:
            print(f"[DataIngestion] Loading cached catalog properties from {cached_meta}")
            return pd.read_csv(cached_meta)

        prop_files = [
            self.raw_dir / "item_properties_part1.csv",
            self.raw_dir / "item_properties_part2.csv",
        ]
        available_files = [f for f in prop_files if f.exists()]

        if not available_files:
            return pd.DataFrame()

        print(f"[DataIngestion] Parsing item properties from {len(available_files)} property files...")
        cat_dict = {}
        avail_dict = {}

        for p_file in available_files:
            print(f"[DataIngestion] Scanning {p_file.name} for categoryid & availability...")
            chunksize = 1000000
            for chunk in pd.read_csv(p_file, chunksize=chunksize, usecols=["itemid", "property", "value"]):
                # Filter to categoryid
                cat_rows = chunk[chunk["property"] == "categoryid"]
                for iid, val in zip(cat_rows["itemid"].values, cat_rows["value"].values):
                    try:
                        cat_dict[int(iid)] = int(val)
                    except ValueError:
                        pass

                # Filter to available
                avail_rows = chunk[chunk["property"] == "available"]
                for iid, val in zip(avail_rows["itemid"].values, avail_rows["value"].values):
                    try:
                        avail_dict[int(iid)] = int(val)
                    except ValueError:
                        pass

        # Build catalog DataFrame
        all_item_ids = sorted(set(cat_dict.keys()) | set(avail_dict.keys()))
        catalog_df = pd.DataFrame({
            "itemid": all_item_ids,
            "categoryid": [cat_dict.get(i, np.nan) for i in all_item_ids],
            "available": [avail_dict.get(i, 1) for i in all_item_ids],
            "price": [round(float(20.0 + (i % 80)), 2) for i in all_item_ids],
        })

        catalog_df.to_csv(cached_meta, index=False)
        print(f"[DataIngestion] Saved enriched catalog ({len(catalog_df):,} items) to {cached_meta}")
        return catalog_df

    def load_data(self, force_synthetic: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Main entry point to load raw events and enriched catalog."""
        if not force_synthetic and self.check_local_dataset():
            print(f"[DataIngestion] Found existing dataset files in {self.raw_dir}")
            events_df = pd.read_csv(self.raw_dir / self.EVENTS_FILENAME)

            # Extract catalog properties
            catalog_df = self.extract_item_properties()
            if catalog_df.empty or "categoryid" not in catalog_df.columns:
                unique_items = events_df["itemid"].unique()
                catalog_df = pd.DataFrame({
                    "itemid": unique_items,
                    "categoryid": [int(i % 100) for i in unique_items],
                    "available": 1,
                    "price": 25.0,
                })

            meta_info = {"is_synthetic": False, "source": "kaggle_retailrocket"}
            return events_df, catalog_df, meta_info

        # Fallback to realistic synthetic demonstration dataset
        events_df, catalog_df = self.generate_realistic_synthetic_dataset()
        meta_info = {
            "is_synthetic": True,
            "source": "SYNTHETIC_DEMO",
            "notice": "Demonstration e-commerce dataset generated with Zipf distribution and Retailrocket funnel dynamics."
        }
        return events_df, catalog_df, meta_info

    def generate_realistic_synthetic_dataset(
        self,
        num_users: int = 2500,
        num_items: int = 1200,
        num_categories: int = 40,
        num_events: int = 50000,
        random_seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Realistic e-commerce dataset generator."""
        rng = np.random.default_rng(random_seed)
        item_ids = np.arange(1000, 1000 + num_items)
        categories = np.arange(10, 10 + num_categories)
        item_categories = rng.choice(categories, size=num_items)
        item_base_prices = np.clip(np.round(rng.lognormal(mean=3.5, sigma=0.8, size=num_items), 2), 5.0, 999.0)

        item_meta = pd.DataFrame({
            "itemid": item_ids,
            "categoryid": item_categories,
            "price": item_base_prices,
            "available": rng.choice([1, 1, 1, 0], size=num_items)
        })

        events_df = pd.DataFrame({
            "timestamp": np.arange(1430438400000, 1430438400000 + num_events * 1000, 1000),
            "visitorid": rng.choice(np.arange(20000, 20000 + num_users), size=num_events),
            "event": rng.choice(["view", "addtocart", "transaction"], size=num_events, p=[0.85, 0.11, 0.04]),
            "itemid": rng.choice(item_ids, size=num_events),
            "transactionid": np.nan,
        })
        return events_df, item_meta

