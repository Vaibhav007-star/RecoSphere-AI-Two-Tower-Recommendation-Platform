"""
A/B Testing Simulation Engine for RecoSphere AI.
Simulates online recommendation experiments comparing Control (Popularity Baseline)
against Treatment (Two-Tower Personalized RecoSphere Engine).
Transparently distinguishes simulated outcomes from empirical historical data.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.experiments.assignment import ExperimentAssignment
from src.experiments.statistics import ExperimentStatistics


class ABExperimentSimulator:
    """
    Simulates e-commerce A/B tests with configurable baseline conversion rates,
    assumed treatment effects (including zero or negative lift), and sample sizes.
    """

    def __init__(self, experiment_id: str = "exp_twotower_retrieval_2026"):
        self.experiment_id = experiment_id
        self.assignment = ExperimentAssignment(experiment_name=experiment_id)

    def run_simulation(
        self,
        sample_size_per_variant: int = 5000,
        baseline_ctr: float = 0.08,  # 8% Click-Through Rate
        baseline_cvr: float = 0.03,  # 3% Conversion Rate (purchases per exposure)
        baseline_addtocart: float = 0.06,  # 6% Add-to-cart Rate
        treatment_lift_ctr: float = 0.18,  # +18% relative lift in CTR (can be <= 0)
        treatment_lift_cvr: float = 0.15,  # +15% relative lift in CVR (can be <= 0)
        treatment_lift_addtocart: float = 0.16,
        avg_order_value_base: float = 45.0,  # $45 base AOV
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes a controlled, reproducible Monte-Carlo experiment simulation.
        Returns detailed metric comparisons, confidence intervals, and hypothesis decisions.
        """
        rng = np.random.default_rng(random_seed)

        n_ctrl = sample_size_per_variant
        n_treat = sample_size_per_variant

        # Effective rates
        rate_ctr_ctrl = baseline_ctr
        rate_ctr_treat = np.clip(baseline_ctr * (1.0 + treatment_lift_ctr), 0.0, 1.0)

        rate_cvr_ctrl = baseline_cvr
        rate_cvr_treat = np.clip(baseline_cvr * (1.0 + treatment_lift_cvr), 0.0, 1.0)

        rate_cart_ctrl = baseline_addtocart
        rate_cart_treat = np.clip(baseline_addtocart * (1.0 + treatment_lift_addtocart), 0.0, 1.0)

        # 1. Simulate Binary Outcomes
        clicks_ctrl = rng.binomial(n=1, p=rate_ctr_ctrl, size=n_ctrl)
        clicks_treat = rng.binomial(n=1, p=rate_ctr_treat, size=n_treat)

        carts_ctrl = rng.binomial(n=1, p=rate_cart_ctrl, size=n_ctrl)
        carts_treat = rng.binomial(n=1, p=rate_cart_treat, size=n_treat)

        conversions_ctrl = rng.binomial(n=1, p=rate_cvr_ctrl, size=n_ctrl)
        conversions_treat = rng.binomial(n=1, p=rate_cvr_treat, size=n_treat)

        # 2. Simulate Continuous Revenue
        # Log-normal order value distribution for converting users
        rev_ctrl = np.zeros(n_ctrl)
        rev_treat = np.zeros(n_treat)

        num_tx_ctrl = int(conversions_ctrl.sum())
        num_tx_treat = int(conversions_treat.sum())

        if num_tx_ctrl > 0:
            tx_orders_ctrl = rng.lognormal(mean=np.log(avg_order_value_base) - 0.2, sigma=0.6, size=num_tx_ctrl)
            rev_ctrl[conversions_ctrl == 1] = np.clip(tx_orders_ctrl, 5.0, 400.0)

        if num_tx_treat > 0:
            # Treatment may also enjoy a slight basket size increase (e.g., +3%)
            treat_aov = avg_order_value_base * 1.03
            tx_orders_treat = rng.lognormal(mean=np.log(treat_aov) - 0.2, sigma=0.6, size=num_tx_treat)
            rev_treat[conversions_treat == 1] = np.clip(tx_orders_treat, 5.0, 400.0)

        # 3. Compute Statistical Tests
        ctr_test = ExperimentStatistics.two_proportion_z_test(
            count_a=int(clicks_ctrl.sum()), n_a=n_ctrl,
            count_b=int(clicks_treat.sum()), n_b=n_treat,
            alpha=alpha,
        )

        cvr_test = ExperimentStatistics.two_proportion_z_test(
            count_a=num_tx_ctrl, n_a=n_ctrl,
            count_b=num_tx_treat, n_b=n_treat,
            alpha=alpha,
        )

        cart_test = ExperimentStatistics.two_proportion_z_test(
            count_a=int(carts_ctrl.sum()), n_a=n_ctrl,
            count_b=int(carts_treat.sum()), n_b=n_treat,
            alpha=alpha,
        )

        revenue_test = ExperimentStatistics.bootstrap_mean_diff_ci(
            samples_a=rev_ctrl,
            samples_b=rev_treat,
            alpha=alpha,
            random_seed=random_seed,
        )

        # 4. Compile Comprehensive Experiment Report
        report = {
            "is_simulation": True,
            "disclaimer": "ALL OUTCOMES ARE SYNTHETICALLY SIMULATED FOR EXPERIMENTAL EVALUATION; NOT LIVE PRODUCTION BUSINESS DATA.",
            "experiment_id": self.experiment_id,
            "sample_size_per_variant": sample_size_per_variant,
            "total_participants": n_ctrl + n_treat,
            "alpha": alpha,
            "random_seed": random_seed,
            "assumptions": {
                "baseline_ctr": baseline_ctr,
                "assumed_lift_ctr": treatment_lift_ctr,
                "baseline_cvr": baseline_cvr,
                "assumed_lift_cvr": treatment_lift_cvr,
                "baseline_addtocart": baseline_addtocart,
                "assumed_lift_addtocart": treatment_lift_addtocart,
                "baseline_aov": avg_order_value_base,
            },
            "metrics": {
                "click_through_rate": ctr_test,
                "conversion_rate": cvr_test,
                "add_to_cart_rate": cart_test,
                "revenue_per_user": revenue_test,
            },
            "sample_data": {
                "control": {
                    "impressions": n_ctrl,
                    "clicks": int(clicks_ctrl.sum()),
                    "carts": int(carts_ctrl.sum()),
                    "purchases": num_tx_ctrl,
                    "total_revenue": round(float(rev_ctrl.sum()), 2),
                    "aov": round(float(rev_ctrl.sum() / max(1, num_tx_ctrl)), 2),
                },
                "treatment": {
                    "impressions": n_treat,
                    "clicks": int(clicks_treat.sum()),
                    "carts": int(carts_treat.sum()),
                    "purchases": num_tx_treat,
                    "total_revenue": round(float(rev_treat.sum()), 2),
                    "aov": round(float(rev_treat.sum() / max(1, num_tx_treat)), 2),
                },
            },
        }

        return report

