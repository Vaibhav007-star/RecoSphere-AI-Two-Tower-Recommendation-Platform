"""
Automated Unit Tests for Data Ingestion, Validation, and Preprocessing.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor
from configs.settings import DataConfig


@pytest.fixture
def sample_events():
    return pd.DataFrame({
        "timestamp": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        "visitorid": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "event": ["view", "addtocart", "view", "transaction", "view", "addtocart", "view", "view", "view", "transaction"],
        "itemid": [101, 101, 102, 102, 103, 104, 105, 106, 107, 108],
        "transactionid": [np.nan, np.nan, np.nan, 999, np.nan, np.nan, np.nan, np.nan, np.nan, 1000],
    })


@pytest.fixture
def sample_catalog():
    return pd.DataFrame({
        "itemid": [101, 102, 103, 104, 105, 106, 107, 108],
        "categoryid": [10, 10, 20, 20, 30, 30, 40, 40],
        "price": [15.0, 25.0, 50.0, 75.0, 99.0, 120.0, 35.0, 45.0],
        "available": [1, 1, 1, 1, 1, 1, 1, 1],
    })


def test_data_validation_valid(sample_events, sample_catalog):
    is_valid, report = DataValidator.validate_events(sample_events)
    assert is_valid is True
    assert report["total_rows"] == 10
    assert report["num_unique_users"] == 5

    is_cat_valid, cat_rep = DataValidator.validate_catalog(sample_catalog)
    assert is_cat_valid is True
    assert cat_rep["total_items"] == 8


def test_data_validation_missing_columns():
    bad_df = pd.DataFrame({"itemid": [1, 2], "event": ["view", "buy"]})
    is_valid, report = DataValidator.validate_events(bad_df)
    assert is_valid is False
    assert len(report["issues"]) > 0


def test_data_leakage_prevention(sample_events, sample_catalog):
    config = DataConfig(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)
    preprocessor = DataPreprocessor(config=config)
    clean_df = preprocessor.clean_events(sample_events)
    train_df, val_df, test_df = preprocessor.split_chronological(clean_df)

    # Train timestamps strictly < val timestamps strictly < test timestamps
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()

    # Encoders fitted on train
    preprocessor.fit_encoders(train_df, sample_catalog)
    transformed_test = preprocessor.transform_events(test_df)

    # User 5 is only in test set -> mapped to -1 (cold user)
    assert 5 not in preprocessor.user_to_idx
    assert (transformed_test[transformed_test["visitorid"] == 5]["user_idx"] == -1).all()

