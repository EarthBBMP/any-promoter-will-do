Test set: n=816

| Promoter input | R² | RMSE | Pearson r |
|---|---|---|---|
| real promoter | 0.330 | 0.391 | 0.598 |
| promoters shuffled across genes | 0.322 | 0.393 | 0.590 |
| promoter zeroed out | -0.076 | 0.495 | 0.590 |

### Movement in predictions when the promoter is destroyed

- promoters shuffled across genes: mean |Δprediction| = 0.05719, max = 0.26752, correlation with real = 0.96655
- promoter zeroed out: mean |Δprediction| = 0.20362, max = 0.66688, correlation with real = 0.92735

### Paired bootstrap: real vs corrupted promoter

| Comparison | ΔR² | 95% CI | p | Verdict |
|---|---|---|---|---|
| real vs promoters shuffled across genes | +0.0079 | [-0.0071, +0.0244] | 0.326 | not significant |
| real vs promoter zeroed out | +0.4052 | [+0.3465, +0.4700] | 0.000 | **significant** |