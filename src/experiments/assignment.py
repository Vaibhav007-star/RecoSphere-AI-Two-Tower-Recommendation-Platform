"""
Consistent Experiment Assignment Module for RecoSphere AI.
Deterministic, reproducible hash-based bucket assignment (SHA-256)
ensuring user consistency across the experiment lifecycle.
"""

import hashlib
from typing import Literal


class ExperimentAssignment:
    """Assigns users deterministically into Control (A) or Treatment (B) variants."""

    def __init__(self, experiment_name: str = "recosphere_twotower_v1", salt: str = "reco_salt_2026"):
        self.experiment_name = experiment_name
        self.salt = salt

    def assign(self, user_id: int, split_ratio: float = 0.50) -> Literal["control", "treatment"]:
        """
        Deterministically maps user_id to 'control' or 'treatment' using SHA-256 hash modulo.
        Guarantees that a user always stays in the same bucket across all sessions.
        """
        key = f"{self.experiment_name}:{self.salt}:{user_id}".encode("utf-8")
        hash_digest = hashlib.sha256(key).hexdigest()
        bucket_val = int(hash_digest[:8], 16) % 10000  # [0, 9999]

        threshold = int(split_ratio * 10000)
        return "control" if bucket_val < threshold else "treatment"

