# Aegis_net ablation (UNSW-NB15, multiclass)

Same split, same 60,000 fit rows, same 3 seeds (42, 43, 44), same protocol as the benchmark. Rarest classes: Worms, Backdoor, Analysis. Every row is a separate training run of the full model with ONE thing changed.

## 1. Resolution of the bins (levels L = log2 of the number of bins per feature; shrinkage 0.03, 4 members)

| Variant | F1 macro raw | std (3 seeds) | F1 macro calibrated | F1 of the 3 rarest classes | note |
|---|---|---|---|---|---|
| levels = 0 | 0.6158 | 0.0070 | 0.6481 | 0.3475 | no bins: only the linear term, i.e. a plain BatchEnsemble MLP |
| levels = 4 | 0.6370 | 0.0027 | 0.6370 | 0.3669 | 16 bins |
| levels = 6 | 0.6810 | 0.0062 | 0.6867 | 0.4846 | 64 bins |
| levels = 8 | 0.6988 | 0.0040 | 0.7023 | 0.5244 | 256 bins (the first MRBE configuration) |
| levels = 10 | 0.7202 | 0.0023 | 0.7287 | 0.5545 | 1,024 bins |
| levels = 12 | 0.7297 | 0.0007 | 0.7366 | 0.5718 | **4,096 bins = the published default** |
| levels = 14 | 0.7373 | 0.0046 | 0.7500 | 0.5868 | 16,384 bins: +0.008 but about 4x slower, not used |

## 2. The other ingredients, switched off one by one at the published configuration (levels = 12)

| Variant | F1 macro raw | std (3 seeds) | F1 macro calibrated | F1 of the 3 rarest classes | note |
|---|---|---|---|---|---|
| full Aegis_net (levels 12, shrink 0.03, 4 members) | 0.7297 | 0.0007 | 0.7366 | 0.5718 | the benchmark configuration |
| no shrinkage (shrink = 0) | 0.6862 | 0.0041 | 0.6899 | 0.5047 | fine bins are not pulled towards their parents |
| 1 member (no ensemble) | 0.7159 | 0.0068 | 0.7315 | 0.5380 |  |

## 3. The same switches measured earlier at levels = 8

| Variant | F1 macro raw | std (3 seeds) | F1 macro calibrated | F1 of the 3 rarest classes | note |
|---|---|---|---|---|---|
| full (levels 8) | 0.6988 | 0.0040 | 0.7023 | 0.5244 |  |
| no shrinkage | 0.6908 | 0.0038 | 0.6886 | 0.5262 |  |
| 1 member | 0.6947 | 0.0021 | 0.6984 | 0.5234 |  |
| 8 members | 0.6997 | 0.0039 | 0.7066 | 0.5245 | no gain over 4 members |

## 4. Strength of the shrinkage (levels = 10)

| Variant | F1 macro raw | std (3 seeds) | F1 macro calibrated | F1 of the 3 rarest classes | note |
|---|---|---|---|---|---|
| shrink = 0.01 | 0.7181 | 0.0019 | 0.7245 | 0.5559 |  |
| shrink = 0.03 | 0.7202 | 0.0023 | 0.7287 | 0.5545 | default |
| shrink = 0.1 | 0.7128 | 0.0022 | 0.7145 | 0.5359 |  |

Reading: the resolution of the bins is the dominant factor (+0.11 F1 macro from no bins to 4,096 bins), shrinkage is what makes very fine bins usable (removing it costs more the finer the bins are), and the ensemble is a smaller, steadier gain. The levels were chosen after looking at this table on the benchmark split; the choice was then confirmed on three other splits (split_check.md).
