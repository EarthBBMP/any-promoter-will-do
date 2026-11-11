"""Regression metrics and bootstrap confidence intervals for ASPINE evaluation.

The original ASPINE repository reported R-squared and RMSE in prose only; no
evaluation code was released. This module makes those numbers reproducible and
adds uncertainty estimates, which a single train/test split cannot otherwise
provide without retraining.
"""

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]


def r_squared(y_true: FloatArray, y_pred: FloatArray) -> float:
    """
    Coefficient of determination, 1 - SS_res / SS_tot.

    Note this is the *variance explained* definition used by Xpresso, not the
    squared Pearson correlation. The two differ whenever predictions are
    biased or mis-scaled, so we report both.

    Examples
    --------
    >>> round(r_squared(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])), 6)
    1.0
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return 1.0 - ss_res / ss_tot


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Root mean squared error."""

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def all_metrics(y_true: FloatArray, y_pred: FloatArray) -> Dict[str, float]:
    """
    Compute the full metric set reported in the paper.

    Returns
    -------
    Dict[str, float]
        r2 (variance explained), rmse, pearson_r, pearson_r2, spearman_r.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # A constant prediction makes correlation undefined; report NaN rather than
    # letting scipy emit a warning and return a spurious value.
    if np.std(y_pred) < 1e-12:
        pearson_r = float("nan")
        spearman_r = float("nan")
    else:
        pearson_r = float(stats.pearsonr(y_true, y_pred)[0])
        spearman_r = float(stats.spearmanr(y_true, y_pred)[0])

    return {
        "r2": r_squared(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson_r": pearson_r,
        "pearson_r2": pearson_r**2,
        "spearman_r": spearman_r,
    }


def bootstrap_ci(
    y_true: FloatArray,
    y_pred: FloatArray,
    metric: str = "r2",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval over the test set.

    Resampling the test genes gives an honest interval on the reported metric
    without retraining the model, which matters here because only one set of
    trained weights was released.

    Parameters
    ----------
    metric : str
        Key from :func:`all_metrics` to bootstrap.
    n_resamples : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided significance level; 0.05 gives a 95% interval.

    Returns
    -------
    Tuple[float, float, float]
        Point estimate on the full test set, lower bound, upper bound.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    point = all_metrics(y_true, y_pred)[metric]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = all_metrics(y_true[idx], y_pred[idx])[metric]

    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

    return point, lower, upper


def paired_bootstrap_delta(
    y_true: FloatArray,
    y_pred_a: FloatArray,
    y_pred_b: FloatArray,
    metric: str = "r2",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Paired bootstrap on the difference between two models' metrics.

    Both models are scored on the *same* resampled genes each iteration, which
    cancels the shared test-set noise that dominates the marginal intervals.
    Without this, two models whose individual CIs overlap heavily can still be
    reliably different (or reliably indistinguishable), and the marginal
    intervals cannot tell you which.

    Returns
    -------
    Dict[str, float]
        delta (a minus b on the full test set), ci_low, ci_high, and p_two_sided,
        the fraction of resamples whose sign disagrees with the point estimate,
        doubled.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_a = np.asarray(y_pred_a, dtype=np.float64).ravel()
    y_pred_b = np.asarray(y_pred_b, dtype=np.float64).ravel()

    delta = all_metrics(y_true, y_pred_a)[metric] - all_metrics(y_true, y_pred_b)[metric]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (
            all_metrics(y_true[idx], y_pred_a[idx])[metric]
            - all_metrics(y_true[idx], y_pred_b[idx])[metric]
        )

    lower = float(np.percentile(deltas, 100 * alpha / 2))
    upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Two-sided bootstrap p-value: how often the resampled difference falls on
    # the opposite side of zero from the observed one.
    if delta >= 0:
        p = 2.0 * float(np.mean(deltas <= 0))
    else:
        p = 2.0 * float(np.mean(deltas >= 0))

    return {
        "delta": delta,
        "ci_low": lower,
        "ci_high": upper,
        "p_two_sided": min(p, 1.0),
    }
