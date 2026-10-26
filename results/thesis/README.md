# Results as reported in the thesis

These files are transcribed from the thesis PDF (`docs/thesis.pdf`, July 2024)
and are kept unchanged as the reference point. The reproduced results on
current library versions are in `results/benchmark/`.

| File | Thesis |
|---|---|
| `performance.csv` | Table 2, accuracy / F1 / precision / recall (single 80/20 split) |
| `roc_auc.csv` | Table 3 |
| `interpretability_ids.csv` | Table 4 |
| `interpretability_mdl.csv` | Table 5 (empty `frac_classes` = not computed in the thesis) |
| `rule_set_size_figure.csv` | Figure 1 (read from the bar chart; `cells` is labelled `wdbc` there) |
| `ids_tuned_lambdas.csv` | Table 6 |
| `ids_tuning_effect.csv` | Table 7 |
| `ruleopt_tuned_params.csv` | Table 8 (ruleopt 1.0 parameter names) |

Empty cells mean the method was not run on that dataset (CORELS only handles
binary targets).
