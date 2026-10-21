"""
Statistical Significance Engine for RecoSphere AI A/B Testing.
Provides:
- Two-proportion Z-tests with p-values
- Absolute and relative lift calculations
- 95% Confidence Intervals for difference in proportions (Wald / Wilson score)
- Non-parametric Bootstrap Confidence Intervals for continuous revenue metrics
- Sample size and statistical power planning calculator
"""

import math
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


class ExperimentStatistics:
    """Computes rigorous statistical hypothesis tests and uncertainty intervals."""

    @staticmethod
    def two_proportion_z_test(
        count_a: int,
        n_a: int,
        count_b: int,
        n_b: int,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Calculates two-proportion Z-test for binary metric (e.g., Conversion Rate).
        Control = A, Treatment = B.
        """
        if n_a == 0 or n_b == 0:
            return {"error": "Sample sizes must be greater than zero."}

        p_a = count_a / n_a
        p_b = count_b / n_b

        # Absolute and Relative Lift
        abs_diff = p_b - p_a
        rel_lift_pct = (abs_diff / p_a * 100) if p_a > 0 else 0.0

        # Pooled proportion under Null Hypothesis H0: p_a == p_b
        p_pool = (count_a + count_b) / (n_a + n_b)
        se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))

        z_stat = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Unpooled Standard Error for Confidence Interval of (p_b - p_a)
        se_diff = math.sqrt((p_a * (1.0 - p_a) / n_a) + (p_b * (1.0 - p_b) / n_b))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = abs_diff - z_crit * se_diff
        ci_upper = abs_diff + z_crit * se_diff

        is_significant = bool(p_value < alpha)

        return {
            "control_rate": round(p_a, 4),
            "treatment_rate": round(p_b, 4),
            "absolute_diff": round(abs_diff, 4),
            "relative_lift_pct": round(rel_lift_pct, 2),
            "z_statistic": round(z_stat, 4),
            "p_value": round(p_value, 5),
            "is_statistically_significant": is_significant,
            "alpha": alpha,
            "confidence_level_pct": int((1 - alpha) * 100),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    @staticmethod
    def bootstrap_mean_diff_ci(
        samples_a: np.ndarray,
        samples_b: np.ndarray,
        num_iterations: int = 1000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Non-parametric bootstrap for skewed continuous metrics like Revenue Per User.
        Computes 95% CI on the difference in means (mean_b - mean_a).
        """
        rng = np.random.default_rng(random_seed)
        mean_a = float(np.mean(samples_a))
        mean_b = float(np.mean(samples_b))
        obs_diff = mean_b - mean_a
        rel_lift_pct = (obs_diff / mean_a * 100) if mean_a > 0 else 0.0

        n_a = len(samples_a)
        n_b = len(samples_b)

        # Vectorized bootstrap
        boot_a = rng.choice(samples_a, size=(num_iterations, n_a), replace=True).mean(axis=1)
        boot_b = rng.choice(samples_b, size=(num_iterations, n_b), replace=True).mean(axis=1)
        boot_diffs = boot_b - boot_a

        low_pct = (alpha / 2) * 100
        high_pct = (1 - alpha / 2) * 100
        ci_lower = float(np.percentile(boot_diffs, low_pct))
        ci_upper = float(np.percentile(boot_diffs, high_pct))

        # Significant if 0 is not inside the CI
        is_significant = bool(ci_lower > 0 or ci_upper < 0)

        # Approximate p-value from bootstrap distribution
        p_val_approx = 2.0 * min(
            (boot_diffs <= 0).mean(),
            (boot_diffs >= 0).mean()
        )

        return {
            "control_mean": round(mean_a, 2),
            "treatment_mean": round(mean_b, 2),
            "absolute_diff": round(obs_diff, 2),
            "relative_lift_pct": round(rel_lift_pct, 2),
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "p_value_approx": round(float(p_val_approx), 4),
            "is_significant": is_significant,
        }

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        mde_relative: float = 0.10,  # Minimum Detectable Effect (e.g. 10% lift)
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """
        Calculates required sample size per variant for two-tailed test of proportions.
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde_relative)
        p_bar = (p1 + p2) / 2.0

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2

        if denominator == 0:
            return 0
        return int(math.ceil(numerator / denominator))

