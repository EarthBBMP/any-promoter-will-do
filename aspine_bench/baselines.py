"""Baseline predictors for the ASPINE expression-prediction task.

The released work reported a single R-squared with nothing to compare it
against. These baselines establish what the number means:

* ``mean``      - the floor. Any model must beat predicting the training mean.
* ``halflife_*``- the 8 tabular mRNA-stability features with no sequence at all.
                  If these match the CNN, the promoter sequence is not
                  contributing and the convolutional stack is unjustified.
* ``kmer_*``    - sequence base composition with no positional information.
                  Separates "the CNN learned regulatory grammar" from "the CNN
                  learned GC content".

Hyperparameters are selected on the validation split, never on test.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FloatArray = npt.NDArray[np.float64]

RIDGE_ALPHAS: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SEED = 0


def predict_mean(train_y: FloatArray, n_test: int) -> FloatArray:
    """Constant predictor at the training mean. Yields test R-squared near 0."""

    return np.full(n_test, float(np.mean(train_y)), dtype=np.float64)


def predict_ridge(
    train_x: FloatArray,
    train_y: FloatArray,
    valid_x: FloatArray,
    valid_y: FloatArray,
    test_x: FloatArray,
) -> Tuple[FloatArray, Dict[str, float]]:
    """
    Ridge regression with alpha chosen on the validation split.

    Features are standardised using training statistics only, so no test
    information reaches the fit.
    """

    scaler = StandardScaler().fit(train_x)
    train_s, valid_s, test_s = (scaler.transform(x) for x in (train_x, valid_x, test_x))

    best_alpha, best_mse = None, np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, random_state=SEED).fit(train_s, train_y)
        mse = float(np.mean((valid_y - model.predict(valid_s)) ** 2))
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse

    model = Ridge(alpha=best_alpha, random_state=SEED).fit(train_s, train_y)

    return model.predict(test_s), {"alpha": float(best_alpha)}


def predict_gbm(
    train_x: FloatArray,
    train_y: FloatArray,
    valid_x: FloatArray,
    valid_y: FloatArray,
    test_x: FloatArray,
) -> Tuple[FloatArray, Dict[str, float]]:
    """
    Gradient-boosted trees, with depth/rate/iterations chosen on the validation
    split.

    sklearn's built-in early stopping carves its monitoring set out of the
    training data, which would ignore the real validation split. Instead we
    scan a small grid and select on the actual validation MSE, matching how
    the ridge baseline is tuned.

    Trees are scale-invariant, so no standardisation is applied.
    """

    grid = [
        {"learning_rate": lr, "max_iter": n, "max_leaf_nodes": leaves}
        for lr in (0.03, 0.1)
        for n in (200, 600)
        for leaves in (15, 31)
    ]

    best_params, best_mse, best_model = None, np.inf, None
    for params in grid:
        model = HistGradientBoostingRegressor(
            early_stopping=False, random_state=SEED, **params
        ).fit(train_x, train_y)
        mse = float(np.mean((valid_y - model.predict(valid_x)) ** 2))
        if mse < best_mse:
            best_params, best_mse, best_model = params, mse, model

    return best_model.predict(test_x), {k: float(v) for k, v in best_params.items()}
