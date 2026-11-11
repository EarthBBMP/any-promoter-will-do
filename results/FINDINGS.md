# ASPINE re-evaluation: findings

Test split: n = 816 genes, label std = 0.4777. Seed 0 throughout.

Reproduce with `python run_benchmark.py --stage all`, `python ablate_sequence.py`,
`python train_cnn.py --seed 0 --seed 1 --seed 2 --optimizer sgd --optimizer adam`,
and `python analyze_retrained.py`.

> **Revision note.** An earlier version of this document concluded that the
> promoter CNN does not outperform eight tabular features. That was measured on
> the *released* weights only. Retraining across seeds shows the released
> weights sit at the low end of what the architecture achieves, and better-trained
> models *do* significantly beat the tabular baseline. The finding below is the
> corrected, more precise one — and it is more interesting.

## 1. The original R² reproduces; the original RMSE does not

| Quantity | Original README | Reproduced |
|---|---|---|
| R² (variance explained) | 0.329 | **0.330** |
| RMSE | 0.440 | **0.391** |

With a test standard deviation of 0.4777, an R² of 0.330 implies
RMSE = 0.4777 × sqrt(1 − 0.330) = 0.391. The reported 0.440 is internally
inconsistent with the reported R². **Report 0.391.**

## 2. The architecture fails to train from scratch half the time

Six cold starts, seeds 0–2 × {SGD (as released), Adam}:

| Seed | Optimizer | Collapsed | R² | Epochs |
|---|---|---|---|---|
| 0 | SGD | **yes** | −0.0001 | 2 |
| 1 | SGD | no | 0.3578 | 17 |
| 2 | SGD | no | 0.3312 | 21 |
| 0 | Adam | **yes** | −0.0005 | 2 |
| 1 | Adam | no | 0.3663 | 14 |
| 2 | Adam | **yes** | −0.0001 | 2 |

**3 of 6 runs collapsed to constant mean-prediction**, with training loss frozen
at exactly the label variance from epoch 2 and both units of the `Dense(2)` ReLU
bottleneck inactive on every input. With no positive activation the gradient
through that layer is identically zero, so the network cannot recover.

Seed 0 collapsed under *both* optimizers and seed 2 collapsed under Adam but not
SGD: the failure is driven by initialisation, and the optimizer modulates but
does not fix it. A 2-unit ReLU layer has no redundancy — if both units die, the
model is dead.

Surviving runs: **R² = 0.3518 ± 0.0184**, which contains the released 0.330.
The published result is reproducible *conditional on training succeeding*.

## 3. A well-trained CNN does beat the tabular baseline

Baselines (all trained on the 13,714-gene training split, tuned on validation):

| Model | R² | 95% CI | RMSE |
|---|---|---|---|
| Train-mean (floor) | −0.001 | [−0.009, −0.000] | 0.478 |
| 6-mer composition only, ridge | −0.038 | [−0.083, 0.001] | 0.487 |
| Half-life features only, ridge | 0.249 | [0.159, 0.324] | 0.414 |
| 6-mer + half-life, ridge | 0.278 | [0.215, 0.333] | 0.406 |
| Half-life features only, GBM | 0.317 | [0.243, 0.382] | 0.395 |
| ASPINE CNN, released weights | 0.330 | [0.271, 0.382] | 0.391 |
| ASPINE CNN, retrained (best) | **0.366** | — | 0.380 |

Paired bootstrap against the 8-feature GBM (2000 resamples, both models scored
on the same resampled genes):

| Model | ΔR² vs GBM | 95% CI | p | Verdict |
|---|---|---|---|---|
| Released weights | +0.0128 | [−0.0167, +0.0454] | 0.385 | not significant |
| Retrained, seed 1 SGD | +0.0409 | [+0.0077, +0.0770] | 0.020 | **significant** |
| Retrained, seed 2 SGD | +0.0143 | [−0.0183, +0.0476] | 0.421 | not significant |
| Retrained, seed 1 Adam | +0.0495 | [+0.0131, +0.0874] | 0.010 | **significant** |
| Mean of 3 survivors | +0.0521 | [+0.0207, +0.0849] | 0.003 | **significant** |

The released weights do not beat eight scalar features. Better-trained models
do, by roughly +0.05 R². **Evaluating only the released checkpoint would have
understated the architecture.**

## 4. But the promoter contributes *non-gene-specific* signal

Intervention on the promoter input, half-life features untouched:

| Model | Real | Promoters shuffled across genes | Promoter zeroed |
|---|---|---|---|
| Released weights | 0.330 | 0.322 | −0.076 |
| Retrained, seed 1 SGD | 0.358 | 0.344 | −0.071 |
| Retrained, seed 2 SGD | 0.331 | 0.328 | −0.286 |
| Retrained, seed 1 Adam | 0.366 | 0.364 | 0.338 |

Paired bootstrap, real vs shuffled promoters:

| Model | ΔR² | 95% CI | p | Verdict |
|---|---|---|---|---|
| Released weights | +0.0079 | [−0.0071, +0.0244] | 0.326 | not significant |
| Retrained, seed 1 Adam | +0.0026 | [−0.0031, +0.0082] | 0.378 | not significant |
| Retrained, seed 2 SGD | +0.0027 | [−0.0128, +0.0185] | 0.741 | not significant |
| Retrained, seed 1 SGD | +0.0134 | [+0.0020, +0.0250] | 0.020 | significant |

Three of four models are statistically indifferent to promoter identity; one
shows a small but significant dependence (+0.013 R²). The effect, where present,
is an order of magnitude smaller than the models' advantage over the tabular
baseline.

Decomposition on the best model (seed 1, Adam):

| Comparison | ΔR² | 95% CI | p | Verdict |
|---|---|---|---|---|
| CNN full promoter vs GBM | +0.0495 | [+0.0131, +0.0874] | 0.010 | **significant** |
| CNN **zeroed** promoter vs GBM | +0.0207 | [−0.0133, +0.0559] | 0.230 | not significant |
| CNN full vs CNN zeroed promoter | +0.0287 | [+0.0034, +0.0515] | 0.031 | **significant** |

**Interpretation.** The promoter branch contributes a real ~0.029 R² — remove it
and the CNN is no longer distinguishable from gradient boosting on eight scalars.
But *which* promoter the model sees does not matter: pairing every gene with a
different gene's real promoter costs +0.0026 (p = 0.378). Any real promoter
serves as well as the correct one.

So the branch is extracting something about the statistics of genuine promoter
sequence — not gene-specific regulatory content. Consistent with the 6-mer
baseline, where composition alone predicts worse than the training mean
(R² = −0.038).

## 5. Framing for the paper

> On reference genomes, promoter sequence is informative for expression
> (Xpresso: r² = 0.59 median, 0.51 cell-type-specific). On this patient-derived,
> WES-based dataset, a CNN reading 10 kb of promoter significantly outperforms
> eight scalar mRNA-stability features (+0.05 R², p = 0.01) — but is statistically
> indifferent to *which gene's* promoter it is shown (p = 0.38). The promoter
> branch contributes signal that is not gene-specific. Separately, the
> architecture fails to train from a cold start in 3 of 6 attempts.

This points at the data-construction pipeline — TSS annotation accuracy and
exome-derived promoter coverage — rather than the architecture, as the binding
constraint on recovering gene-specific regulatory signal.

## 6. Limitations to state explicitly

- Splits are random over genes, not chromosome-held-out. Gene IDs are disjoint
  across splits (verified: zero overlap between every pair), but paralogues and
  bidirectional promoters can still leak at the sequence level.
- Three seeds × two optimizers is a small sweep. The 50% collapse rate is an
  estimate from six runs, not a precise figure.
- Confidence intervals are bootstrap over test genes; they describe test-set
  uncertainty, and are reported separately from across-seed variation.
- Hyperparameters were inherited from Xpresso, tuned there for a 10,500 bp
  input; the model consumes 10,000 bp and was not re-tuned.
- Only the full-length patient-data configuration (Experiment 3) is evaluated;
  weights for the other three reported experiments were never released.
- Three of 13,714 training genes (0.02%) are stored shorter than the nominal
  20 kb window and are zero-padded, as the original pipeline did silently.
  Validation and test splits are complete.
