"""
Automated Unit Tests for A/B Testing Assignment, Statistical Calculations, and Simulation.
"""

import pytest
import numpy as np
from src.experiments.assignment import ExperimentAssignment
from src.experiments.statistics import ExperimentStatistics
from src.experiments.simulator import ABExperimentSimulator


def test_consistent_experiment_assignment():
    assignment = ExperimentAssignment(experiment_name="test_exp")
    # Same user should always receive the same group assignment
    variant_1 = assignment.assign(user_id=12345)
    variant_2 = assignment.assign(user_id=12345)
    assert variant_1 == variant_2
    assert variant_1 in ["control", "treatment"]


def test_two_proportion_z_test():
    # Significant difference test
    res = ExperimentStatistics.two_proportion_z_test(
        count_a=100, n_a=2000,  # 5%
        count_b=160, n_b=2000,  # 8%
        alpha=0.05,
    )
    assert res["control_rate"] == 0.05
    assert res["treatment_rate"] == 0.08
    assert res["relative_lift_pct"] == 60.0
    assert res["is_statistically_significant"] is True
    assert res["p_value"] < 0.05
    assert res["ci_lower"] > 0.0


def test_bootstrap_revenue_ci():
    rng = np.random.default_rng(42)
    control_rev = rng.normal(loc=10.0, scale=2.0, size=100)
    treat_rev = rng.normal(loc=15.0, scale=2.0, size=100)

    res = ExperimentStatistics.bootstrap_mean_diff_ci(
        samples_a=control_rev,
        samples_b=treat_rev,
        num_iterations=200,
        alpha=0.05,
    )
    assert res["is_significant"] is True
    assert res["ci_lower"] > 0.0


def test_ab_simulator():
    sim = ABExperimentSimulator()
    result = sim.run_simulation(
        sample_size_per_variant=500,
        baseline_ctr=0.05,
        treatment_lift_ctr=0.20,
    )
    assert result["is_simulation"] is True
    assert "metrics" in result
    assert "click_through_rate" in result["metrics"]
    assert result["total_participants"] == 1000

