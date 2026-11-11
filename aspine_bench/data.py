"""Memory-efficient loading of the ASPINE dataset.

Two differences from the original ``utils.encode_sequence``:

1. Sequences are sliced to the model's input window *before* one-hot encoding,
   not after. The released pipeline allocated the full 20,000 bp encoding and
   then discarded half of it, costing ~4.4 GB for the training split.
2. The one-hot tensor is built with a lookup table into ``int8`` rather than a
   Python double loop into ``float32``, a further 4x reduction and roughly two
   orders of magnitude faster. Callers cast to float at the batch level.

Together these let the full training split be encoded in ~0.55 GB instead of
~4.4 GB, which is the difference between needing a workstation and running on
a free Colab instance.
"""

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

# The 8 mRNA-stability ("half-life") features, in the order the released model
# was trained with. Order matters: the saved weights assume this layout.
HALFLIFE_FEATURES: List[str] = [
    "utr5_length",
    "cds_length",
    "intron_length",
    "utr3_length",
    "utr5_gc",
    "cds_gc",
    "utr3_gc",
    "orf_exon_density",
]

LABEL = "MPE"

# Stored sequences span TSS +/- 10,000 bp. The released model consumes
# [3500:13500], i.e. -6,500 to +3,500 relative to the TSS.
WINDOW: Tuple[int, int] = (3500, 13500)

# Maps ASCII byte -> base index; 255 marks "not a base" (N and anything else).
_BASE_LUT = np.full(256, 255, dtype=np.uint8)
for _char, _idx in [("A", 0), ("C", 1), ("G", 2), ("T", 3)]:
    _BASE_LUT[ord(_char)] = _idx
    _BASE_LUT[ord(_char.lower())] = _idx


def load_tabular(path: str) -> pd.DataFrame:
    """
    Load labels and half-life features, skipping the sequence column entirely.

    The SEQ column dominates file size (277 MB for the training split), so
    baselines that never touch sequence should not pay to materialise it.
    """

    df = pd.read_csv(path, usecols=["gene_id", LABEL] + HALFLIFE_FEATURES)

    return df.dropna().reset_index(drop=True)


def _to_codes(sequence: str, window: Optional[Tuple[int, int]]) -> npt.NDArray[np.uint8]:
    """Slice a sequence to the window, then map bases to 0-3 (255 = ambiguous)."""

    if window is not None:
        sequence = sequence[window[0] : window[1]]

    return _BASE_LUT[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def one_hot(
    sequences: Sequence[str],
    window: Optional[Tuple[int, int]] = WINDOW,
    dtype: npt.DTypeLike = np.int8,
    pad_to: Optional[int] = None,
) -> npt.NDArray:
    """
    One-hot encode sequences over the model input window.

    Ambiguous bases (N) become all-zero columns, matching the released
    ``utils.encode_sequence`` behaviour.

    By default every sequence must have the same length after slicing. The
    original sized the whole batch from the *first* sequence, so a shorter
    sequence was silently zero-padded and a longer one raised ``IndexError`` --
    a difference in input the caller never saw.

    Three of the 13,714 training genes are stored shorter than the nominal
    20,000 bp (3,612 to 8,773 bp), so the original pipeline was silently
    zero-padding their promoter tails. Passing ``pad_to`` reproduces that
    padding deliberately: short sequences are right-padded with all-zero
    columns and long ones truncated, so the behaviour is explicit rather than
    an accident of ordering. Callers that pad should report how many sequences
    they padded.
    """

    codes = [_to_codes(s, window) for s in sequences]

    if pad_to is None:
        lengths = {len(c) for c in codes}
        if len(lengths) != 1:
            raise ValueError(
                f"Sequences have inconsistent lengths after slicing: {sorted(lengths)}. "
                f"Pass pad_to={max(lengths)} to zero-pad, as the original pipeline did."
            )
        length = lengths.pop()
    else:
        length = pad_to

    out = np.zeros((len(codes), length, 4), dtype=dtype)

    for i, code in enumerate(codes):
        code = code[:length]
        known = code != 255
        out[i, np.flatnonzero(known), code[known]] = 1

    return out


def count_short_sequences(
    sequences: Sequence[str], window: Tuple[int, int] = WINDOW
) -> int:
    """Number of sequences that do not fill the window and will be zero-padded."""

    expected = window[1] - window[0]

    return sum(1 for s in sequences if len(_to_codes(s, window)) < expected)


def kmer_frequencies(
    sequences: Sequence[str],
    k: int = 6,
    window: Optional[Tuple[int, int]] = WINDOW,
) -> npt.NDArray[np.float32]:
    """
    Normalised k-mer frequency vector per sequence, shape (n, 4**k).

    This is the feature set for the sequence-composition baseline: it captures
    what a model could learn from base composition alone, with no positional
    information. If the CNN cannot beat it, the convolutional architecture is
    not earning its input.

    Windows containing an ambiguous base are skipped rather than counted.
    """

    n_features = 4**k
    out = np.zeros((len(sequences), n_features), dtype=np.float32)

    for i, sequence in enumerate(sequences):
        codes = _to_codes(sequence, window).astype(np.int64)
        n_windows = len(codes) - k + 1
        if n_windows <= 0:
            continue

        valid = np.ones(n_windows, dtype=bool)
        index = np.zeros(n_windows, dtype=np.int64)

        # Roll the k-mer index forward one base at a time (base-4 digits).
        for j in range(k):
            base = codes[j : j + n_windows]
            valid &= base != 255
            index = index * 4 + np.where(base == 255, 0, base)

        counts = np.bincount(index[valid], minlength=n_features).astype(np.float32)
        total = counts.sum()
        if total > 0:
            out[i] = counts / total

    return out


def iter_sequence_chunks(
    path: str, chunksize: int = 512
) -> Iterator[Tuple[List[str], pd.DataFrame]]:
    """
    Stream (sequences, tabular) pairs so the 277 MB training split never has to
    be held in memory in full.
    """

    columns = ["gene_id", LABEL, "SEQ"] + HALFLIFE_FEATURES

    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk = chunk.dropna()
        if chunk.empty:
            continue
        yield chunk["SEQ"].tolist(), chunk.drop(columns=["SEQ"]).reset_index(drop=True)
