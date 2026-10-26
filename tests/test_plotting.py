import pandas as pd

from xai_rules import plotting


def _summary():
    rows = []
    for ds in ["cells", "glass"]:
        for i, m in enumerate(["ruleopt", "mdl", "ids", "corels", "cart", "c45", "random_forest"]):
            if ds == "glass" and m == "corels":
                continue  # skipped models are simply absent from the summary
            rule = m != "random_forest"
            rows.append(
                {
                    "dataset": ds,
                    "model": m,
                    "accuracy_mean": 0.7 + 0.03 * i,
                    "accuracy_std": 0.01,
                    "roc_auc_mean": 0.8,
                    "roc_auc_std": 0.01,
                    "n_rules_mean": 5.0 + i if rule else float("nan"),
                    "avg_rule_width_mean": 2.0 if rule else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def test_all_figures_render(tmp_path):
    s = _summary()
    thesis = pd.DataFrame({"metric": ["accuracy"], "dataset": ["cells"], "mdl": [0.94], "cart": [0.9]})
    paths = [
        plotting.accuracy_heatmap(s, tmp_path / "a.png"),
        plotting.accuracy_heatmap(s, tmp_path / "b.png", metric="roc_auc"),
        plotting.accuracy_vs_rules(s, tmp_path / "c.png"),
        plotting.rule_set_shape(s, tmp_path / "d.png"),
        plotting.thesis_vs_reproduced(s, thesis, tmp_path / "e.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 10_000
