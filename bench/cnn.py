"""Evaluation of the released ASPINE convolutional model.

The published repository ships one set of trained weights, whose input layer is
(None, 10000, 4) - the full-length patient-data configuration. Evaluating it
requires only a forward pass, so no retraining is involved and the numbers are
deterministic.
"""

import os
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

# Keep TensorFlow quiet; it logs device banners to stderr on import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from bench import config, data


def load_released_model():
    """Load the released .keras model, weights included."""

    from tensorflow.keras.models import load_model

    return load_model(str(config.WEIGHTS_PATH), compile=False)


def expected_promoter_length(model) -> int:
    """Input window length the released weights expect."""

    for layer in model.inputs:
        if len(layer.shape) == 3:
            return int(layer.shape[1])
    raise ValueError("No 3-D promoter input found on the model.")


def predict(
    model,
    sequences: List[str],
    halflife: npt.NDArray[np.float64],
    batch_size: int = 64,
) -> npt.NDArray[np.float64]:
    """
    Run the model over sequences, encoding in batches.

    Encoding is done per batch and cast to float32 only at the last moment, so
    peak memory stays proportional to the batch rather than the split.
    """

    outputs: List[npt.NDArray[np.float32]] = []

    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        promoter = data.one_hot(chunk).astype(np.float32)
        features = halflife[start : start + batch_size].astype(np.float32)
        outputs.append(model.predict([promoter, features], verbose=0).ravel())

    return np.concatenate(outputs).astype(np.float64)
