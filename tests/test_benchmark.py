"""Tests for the benchmark utilities.

Note the error-handling style: the original repository's test wrapped the
failing call in ``try/except`` around a ``print``, so a call that raised
*nothing* silently passed. ``pytest.raises`` makes the absence of an exception
a failure, which is the point of the test.
"""

import numpy as np
import pytest

from aspine_bench import data, metrics


def test_one_hot_matches_original_semantics():
    encoded = data.one_hot(["ACGT"], window=None)
    expected = np.array(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.int8
    )
    assert np.array_equal(encoded, expected)


def test_one_hot_treats_n_as_all_zero():
    encoded = data.one_hot(["ACNT"], window=None)
    assert np.array_equal(encoded[0, 2], np.zeros(4, dtype=np.int8))


def test_one_hot_is_case_insensitive():
    assert np.array_equal(
        data.one_hot(["acgt"], window=None), data.one_hot(["ACGT"], window=None)
    )


def test_one_hot_rejects_ragged_input():
    # The original sized the whole batch from the first sequence, silently
    # zero-padding shorter ones and raising IndexError on longer ones.
    with pytest.raises(ValueError):
        data.one_hot(["ACGT", "AC"], window=None)


def test_one_hot_slices_before_encoding():
    encoded = data.one_hot(["AAAACCCC"], window=(4, 8))
    assert encoded.shape == (1, 4, 4)
    assert encoded[0, :, 1].sum() == 4


def test_kmer_frequencies_sum_to_one():
    freqs = data.kmer_frequencies(["ACGTACGTAC"], k=3, window=None)
    assert freqs.shape == (1, 64)
    assert np.isclose(freqs.sum(), 1.0)


def test_kmer_frequencies_skip_ambiguous_windows():
    clean = data.kmer_frequencies(["ACGTACGT"], k=2, window=None)
    with_n = data.kmer_frequencies(["ACGTNCGT"], k=2, window=None)
    assert not np.allclose(clean, with_n)
    assert np.isclose(with_n.sum(), 1.0)


def test_r_squared_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(metrics.r_squared(y, y), 1.0)
    assert np.isclose(metrics.r_squared(y, np.full_like(y, y.mean())), 0.0)


def test_rmse_matches_manual_computation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert np.isclose(metrics.rmse(y_true, y_pred), np.sqrt(4.0 / 3.0))


def test_all_metrics_handles_constant_prediction():
    y = np.array([1.0, 2.0, 3.0])
    result = metrics.all_metrics(y, np.full_like(y, 2.0))
    assert np.isnan(result["pearson_r"])
    assert np.isclose(result["r2"], 0.0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    preds = y + rng.normal(scale=0.5, size=400)
    point, low, high = metrics.bootstrap_ci(y, preds, "r2", n_resamples=200, seed=0)
    assert low < point < high


def test_paired_bootstrap_detects_no_difference_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    preds = y + rng.normal(scale=0.5, size=300)
    result = metrics.paired_bootstrap_delta(y, preds, preds, "r2", n_resamples=200, seed=0)
    assert np.isclose(result["delta"], 0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_one_hot_pads_short_sequences_when_asked():
    # The released training split contains three genes shorter than the
    # nominal 20,000 bp; the original pipeline zero-padded them silently.
    encoded = data.one_hot(["ACGT", "AC"], window=None, pad_to=4)
    assert encoded.shape == (2, 4, 4)
    assert encoded[1, 0, 0] == 1 and encoded[1, 1, 1] == 1
    assert encoded[1, 2].sum() == 0 and encoded[1, 3].sum() == 0


def test_one_hot_truncates_long_sequences_when_padding():
    encoded = data.one_hot(["ACGTACGT"], window=None, pad_to=4)
    assert encoded.shape == (1, 4, 4)
    assert np.array_equal(encoded, data.one_hot(["ACGT"], window=None))


def test_one_hot_error_message_suggests_pad_to():
    with pytest.raises(ValueError, match="pad_to"):
        data.one_hot(["ACGT", "AC"], window=None)


def test_count_short_sequences():
    full = "A" * 20000
    short = "A" * 9000
    assert data.count_short_sequences([full, full]) == 0
    assert data.count_short_sequences([full, short]) == 1
