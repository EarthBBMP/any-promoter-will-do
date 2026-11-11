| Model | R² (var. explained) | 95% CI | RMSE | Pearson r | Pearson r² | Spearman ρ |
|---|---|---|---|---|---|---|
| Train-mean (floor) | -0.001 | [-0.009, -0.000] | 0.478 | nan | nan | nan |
| Half-life features only, ridge | 0.249 | [0.159, 0.324] | 0.414 | 0.629 | 0.395 | 0.592 |
| Half-life features only, GBM | 0.317 | [0.243, 0.382] | 0.395 | 0.619 | 0.383 | 0.571 |
| 6-mer composition only, ridge | -0.038 | [-0.083, 0.001] | 0.487 | 0.074 | 0.005 | 0.054 |
| 6-mer + half-life, ridge | 0.278 | [0.215, 0.333] | 0.406 | 0.559 | 0.312 | 0.496 |
| ASPINE CNN (released weights) | 0.330 | [0.271, 0.382] | 0.391 | 0.598 | 0.358 | 0.557 |