"""
Data Validation and Quality Assurance Module for RecoSphere AI.
Validates raw and processed interaction datasets against required schemas and business rules.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


class DataValidator:
    """Performs schema, range, and consistency validation on interaction and catalog data."""

    REQUIRED_EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid"]
    VALID_EVENT_TYPES = {"view", "addtocart", "transaction"}

    @classmethod
    def validate_events(cls, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates the events DataFrame.
        Returns (is_valid, validation_report).
        """
        issues: List[str] = []
        warnings: List[str] = []

        # 1. Column existence
        missing_cols = [c for c in cls.REQUIRED_EVENT_COLUMNS if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        if issues:
            return False, {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "row_count": len(df),
            }

        # 2. Row count
        if len(df) == 0:
            issues.append("Dataset is empty.")
            return False, {"is_valid": False, "issues": issues, "warnings": warnings, "row_count": 0}

        # 3. Missing values
        null_counts = df[cls.REQUIRED_EVENT_COLUMNS].isnull().sum().to_dict()
        for col, count in null_counts.items():
            if count > 0:
                issues.append(f"Column '{col}' contains {count} missing values.")

        # 4. Event types
        observed_events = set(df["event"].unique())
        invalid_events = observed_events - cls.VALID_EVENT_TYPES
        if invalid_events:
            issues.append(f"Unrecognized event types found: {invalid_events}")

        # 5. Timestamp validation
        if not pd.api.types.is_numeric_dtype(df["timestamp"]) and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            issues.append("Timestamp must be numeric (epoch ms/s) or datetime.")

        # 6. Duplicate records check
        duplicates = df.duplicated(subset=["timestamp", "visitorid", "itemid", "event"]).sum()
        if duplicates > 0:
            warnings.append(f"Detected {duplicates} exact duplicate interaction rows (can be deduplicated).")

        # 7. Statistics
        num_users = df["visitorid"].nunique()
        num_items = df["itemid"].nunique()
        event_distribution = df["event"].value_counts().to_dict()

        report = {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_rows": int(len(df)),
            "num_unique_users": int(num_users),
            "num_unique_items": int(num_items),
            "event_distribution": event_distribution,
            "duplicates_count": int(duplicates),
            "timestamp_min": str(df["timestamp"].min()),
            "timestamp_max": str(df["timestamp"].max()),
        }

        return len(issues) == 0, report

    @classmethod
    def validate_catalog(cls, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates product catalog metadata (item properties, category, price, etc.).
        """
        issues: List[str] = []
        warnings: List[str] = []

        if "itemid" not in df.columns:
            issues.append("Catalog missing primary key 'itemid'.")

        if issues:
            return False, {"is_valid": False, "issues": issues, "warnings": warnings}

        null_items = df["itemid"].isnull().sum()
        if null_items > 0:
            issues.append(f"Found {null_items} null item IDs.")

        duplicate_items = df.duplicated(subset=["itemid"]).sum()
        if duplicate_items > 0:
            warnings.append(f"Found {duplicate_items} duplicate item IDs in catalog.")

        return len(issues) == 0, {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_items": int(df["itemid"].nunique()),
            "columns": list(df.columns),
        }

