"""
Execution script to acquire, validate, preprocess, and analyze dataset for RecoSphere AI.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.eda import DataAnalyzer


def main():
    print("=" * 70)
    print("RecoSphere AI — Phase 2: Data Ingestion & Preprocessing Pipeline")
    print("=" * 70)

    # 1. Ingestion
    ingestion = DataIngestion()
    events_df, catalog_df, meta_info = ingestion.load_data()
    print(f"[Main] Ingestion Source: {meta_info.get('source')} (is_synthetic: {meta_info.get('is_synthetic')})")
    print(f"[Main] Loaded {len(events_df):,} events and {len(catalog_df):,} catalog items.")

    # 2. Preprocessing & Chronological Splitting
    preprocessor = DataPreprocessor()
    processed_dict = preprocessor.run_pipeline(events_df, catalog_df)

    # 3. Exploratory Data Analysis
    analyzer = DataAnalyzer()
    eda_report = analyzer.analyze(
        events_df=events_df,
        catalog_df=catalog_df,
        train_df=processed_dict["train_df"],
        val_df=processed_dict["val_df"],
        test_df=processed_dict["test_df"],
    )

    print("\n[Main] Data Preparation Summary:")
    print(f" - Train interactions: {len(processed_dict['train_df']):,}")
    print(f" - Val interactions:   {len(processed_dict['val_df']):,}")
    print(f" - Test interactions:  {len(processed_dict['test_df']):,}")
    print(f" - Unique Users:       {eda_report['total_users']:,}")
    print(f" - Unique Items:       {eda_report['total_items']:,}")
    print(f" - Matrix Sparsity:    {eda_report['matrix_sparsity_pct']}%")
    print(f" - Pareto Top 20% Vol: {eda_report['pareto_top20_item_event_share_pct']}%")
    print("\nPhase 2 completed successfully!")


if __name__ == "__main__":
    main()

