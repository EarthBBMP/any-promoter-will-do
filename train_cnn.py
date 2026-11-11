"""Retrain the ASPINE CNN from scratch, sweeping seeds and optimizers.

This is optional. Every finding in ``results/FINDINGS.md`` is measured on the
*released* weights with a forward pass, and does not depend on retraining.

KNOWN FAILURE MODE. Retraining from scratch under the released optimizer
settings (SGD, lr=5e-4, momentum=0.9) collapsed to constant mean-prediction on
seed 0: training loss froze at 0.2301 (exactly the training-set variance) from
epoch 2 onward, and inspection of the checkpoint showed both units of the
``Dense(2)`` ReLU bottleneck inactive on every input. With no positive
activation the gradient through that layer is identically zero, so the network
cannot recover and the final ``Dense(1)`` emits only its bias.

Rather than wait out 250 dead epochs, ``DeadRunDetector`` watches the model's
output on a held-out probe batch and stops as soon as predictions go constant.
That makes it cheap to ask the question this failure actually raises: *how
often* does the released architecture collapse? Sweeping seeds and optimizers
turns a broken run into a measured reproducibility rate.

    python train_cnn.py --seed 0 --seed 1 --seed 2
    python train_cnn.py --seed 0 --seed 1 --optimizer sgd --optimizer adam
    python train_cnn.py --seed 0 --max-epochs 40

Cost on CPU: ~76 s/epoch over the 13,714-gene training split, encoding ~6 s.
On a Colab T4/A100 expect roughly 5-15x faster, which is what makes a sweep
practical. Results land in results/retrained.csv.
"""

import argparse
import json
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from keras.callbacks import Callback
from keras.utils import PyDataset

from bench import config, data, metrics


class EncodedSequence(PyDataset):
    """
    Keras data pipeline that holds promoters as ``int8`` and casts per batch.

    Holding the training split one-hot encoded as float32 would need ~2.2 GB.
    Stored as int8 it is ~0.55 GB, and each batch is cast to float32 only as
    the model consumes it, which costs nothing measurable next to the
    convolutions.

    Must subclass ``PyDataset``: Keras 3 rejects a plain object implementing
    ``__len__``/``__getitem__`` with "Unrecognized data type".
    """

    def __init__(
        self,
        promoter: npt.NDArray[np.int8],
        halflife: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.promoter = promoter
        self.halflife = halflife
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(targets))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return int(np.ceil(len(self.targets) / self.batch_size))

    def __getitem__(self, index: int):
        idx = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        # The two inputs must be a tuple, not a list: Keras 3 builds a
        # tf.TypeSpec from whatever __getitem__ returns and rejects a list.
        return (
            (self.promoter[idx].astype(np.float32), self.halflife[idx]),
            self.targets[idx],
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)


class DeadRunDetector(Callback):
    """
    Stop training once the network's output collapses to a constant.

    Watching the loss is not enough: a dead run's loss sits at exactly the
    label variance, which looks like a plateau rather than a failure. Measuring
    the spread of predictions on a fixed probe batch names the failure directly
    -- zero variance means every gene receives the same prediction.
    """

    def __init__(self, probe, patience: int = 2, tol: float = 1e-6) -> None:
        super().__init__()
        self.probe = probe
        self.patience = patience
        self.tol = tol
        self.consecutive = 0
        self.collapsed = False
        self.collapsed_epoch: Optional[int] = None

    def on_epoch_end(self, epoch, logs=None) -> None:
        preds = self.model.predict(self.probe, verbose=0).ravel()
        spread = float(preds.std())

        if spread < self.tol:
            self.consecutive += 1
        else:
            self.consecutive = 0

        if self.consecutive >= self.patience:
            self.collapsed = True
            self.collapsed_epoch = epoch + 1
            self.model.stop_training = True
            print(
                f"    collapsed: prediction std = {spread:.2e} for "
                f"{self.patience} consecutive epochs; stopping."
            )


def encode_split(csv_path) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Stream a split from disk and one-hot encode it into an int8 array.

    Sequences are padded to the full window rather than required to fill it:
    three training genes are stored shorter than 20,000 bp, and the original
    pipeline zero-padded them silently. We do the same thing on purpose and
    report the count.
    """

    width = data.WINDOW[1] - data.WINDOW[0]
    promoters: List[npt.NDArray[np.int8]] = []
    tables: List[pd.DataFrame] = []
    n_padded = 0

    for sequences, table in data.iter_sequence_chunks(str(csv_path)):
        n_padded += data.count_short_sequences(sequences)
        promoters.append(data.one_hot(sequences, dtype=np.int8, pad_to=width))
        tables.append(table)

    promoter = np.concatenate(promoters)
    table = pd.concat(tables, ignore_index=True)

    if n_padded:
        print(
            f"  note: {n_padded} sequence(s) in {csv_path.name} were shorter than "
            f"{width} bp and were zero-padded (as the original pipeline did silently)"
        )

    return (
        promoter,
        table[data.HALFLIFE_FEATURES].to_numpy(np.float32),
        table[data.LABEL].to_numpy(np.float32),
    )


def build_optimizer(name: str, lr: Optional[float]):
    """Released settings are SGD(5e-4, momentum=0.9); adam is offered for comparison."""

    from keras.optimizers import SGD, Adam

    if name == "sgd":
        return SGD(learning_rate=lr if lr is not None else 5e-4, momentum=0.9)
    if name == "adam":
        return Adam(learning_rate=lr if lr is not None else 1e-3)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one(
    seed: int,
    optimizer: str,
    lr: Optional[float],
    max_epochs: int,
    batch_size: int,
    splits,
) -> dict:
    """Train a single model and return its test-set metrics."""

    import tensorflow as tf
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    sys.path.insert(0, str(config.REPO_ROOT))
    from xpresso_module.model import XPressoModel

    # Seed every source of randomness the run touches.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    (train_p, train_h, train_y), (valid_p, valid_h, valid_y), (test_p, test_h, test_y) = splits

    model = XPressoModel(promoter_shape=train_p.shape[1:], halflife_shape=train_h.shape[1:])
    model._model.compile(
        optimizer=build_optimizer(optimizer, lr),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )

    tag = f"seed{seed}_{optimizer}"
    checkpoint = config.RESULTS_DIR / f"retrained_{tag}.keras"

    train_gen = EncodedSequence(train_p, train_h, train_y, batch_size, True, seed)
    valid_gen = EncodedSequence(valid_p, valid_h, valid_y, batch_size, False, seed)
    test_gen = EncodedSequence(test_p, test_h, test_y, batch_size, False, seed)

    detector = DeadRunDetector(valid_gen)
    callbacks = [
        detector,
        EarlyStopping(monitor="val_loss", patience=7, mode="min", restore_best_weights=True),
        ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True, mode="min"),
    ]

    started = time.time()
    history = model._model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - started

    preds = model._model.predict(test_gen, verbose=0).ravel().astype(np.float64)
    np.save(config.RESULTS_DIR / f"preds_retrained_{tag}.npy", preds)
    (config.RESULTS_DIR / f"history_{tag}.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2)
    )

    result = metrics.all_metrics(test_y.astype(np.float64), preds)
    result.update(
        {
            "seed": seed,
            "optimizer": optimizer,
            "lr": lr,
            "collapsed": detector.collapsed,
            "pred_std": float(preds.std()),
            "epochs_run": len(history.history["loss"]),
            "minutes": elapsed / 60.0,
        }
    )
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument(
        "--optimizer", choices=["sgd", "adam"], action="append", dest="optimizers"
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    seeds = args.seeds or [0]
    optimizers = args.optimizers or ["sgd"]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Encoding splits (int8; done once and reused across every run)")
    started = time.time()
    splits = (
        encode_split(config.TRAIN_CSV),
        encode_split(config.VALID_CSV),
        encode_split(config.TEST_CSV),
    )
    print(
        f"  train {splits[0][0].shape} = {splits[0][0].nbytes / 1e9:.2f} GB as int8"
        f"  ({time.time() - started:.0f}s)"
    )

    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            print(f"\n{'=' * 60}\nseed {seed}  |  optimizer {optimizer}\n{'=' * 60}")
            row = train_one(seed, optimizer, args.lr, args.max_epochs, args.batch_size, splits)
            rows.append(row)
            state = "COLLAPSED" if row["collapsed"] else "trained"
            print(
                f"  -> {state}: R² = {row['r2']:.4f}  pred_std = {row['pred_std']:.5f}"
                f"  ({row['epochs_run']} epochs, {row['minutes']:.0f} min)"
            )
            # Write after every run so a disconnect never loses finished work.
            pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "retrained.csv", index=False)

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}\nSummary ({len(df)} run(s))\n{'=' * 60}")
    print(
        df[["seed", "optimizer", "collapsed", "r2", "rmse", "pred_std", "epochs_run", "minutes"]]
        .to_string(index=False)
    )

    n_collapsed = int(df["collapsed"].sum())
    print(f"\ncollapsed to constant output: {n_collapsed}/{len(df)} runs")

    alive = df[~df["collapsed"]]
    if len(alive) > 1:
        print(
            f"R² across surviving runs: {alive['r2'].mean():.4f} ± {alive['r2'].std(ddof=1):.4f}"
            f"   (released weights: 0.330)"
        )
    elif len(alive) == 1:
        print(f"R² of the one surviving run: {alive['r2'].iloc[0]:.4f}   (released weights: 0.330)")
    else:
        print("No run escaped collapse. Report this as a reproducibility finding.")


if __name__ == "__main__":
    main()
