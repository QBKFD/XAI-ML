"""Figures for the README. All take the ``summary.csv`` produced by the benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from xai.models import MODELS, RULE_LEARNERS  # noqa: E402

# Palette: categorical slots in fixed order, checked for colour-vision
# deficiency separation; one blue ramp for magnitudes; neutral ink for text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e6e5e1"
MISSING = "#efeeea"

DATASET_ORDER = [
    "cells",
    "titanic",
    "tictactoe",
    "sonar",
    "ionosphere",
    "liver",
    "compas",
    "cars",
    "glass",
    "doctor",
    "drybeans",
]
MODEL_ORDER = list(MODELS)
NAMES = {k: cls.display_name for k, cls in MODELS.items()}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK_3,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
        }
    )


def _datasets(summary: pd.DataFrame) -> list[str]:
    present = set(summary["dataset"])
    return [d for d in DATASET_ORDER if d in present] + sorted(present - set(DATASET_ORDER))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def accuracy_heatmap(summary: pd.DataFrame, path: Path, metric: str = "accuracy") -> Path:
    """Datasets x models, mean test score over seeds."""
    _style()
    datasets = _datasets(summary)
    models = [m for m in MODEL_ORDER if m in set(summary["model"])]
    grid = summary.pivot(index="dataset", columns="model", values=f"{metric}_mean").reindex(
        index=datasets, columns=models
    )
    values = grid.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.05 * len(models) + 2.2, 0.42 * len(datasets) + 1.4))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    lo = np.floor(np.nanmin(values) * 10) / 10
    masked = np.ma.masked_invalid(values)
    cmap.set_bad(MISSING)
    ax.imshow(masked, cmap=cmap, vmin=lo, vmax=1.0, aspect="auto")

    for i in range(values.shape[0]):
        best = np.nanmax(values[i]) if np.isfinite(values[i]).any() else None
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", color=INK_3, fontsize=8.5)
                continue
            dark = (v - lo) / (1 - lo) > 0.55
            weight = "bold" if best is not None and np.isclose(v, best) else "normal"
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if dark else INK,
                fontweight=weight,
            )

    ax.set_xticks(range(len(models)), [NAMES[m] for m in models])
    ax.set_yticks(range(len(datasets)), datasets)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White separators between cells and between rule learners and baselines.
    ax.set_xticks(np.arange(-0.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    n_rule = sum(m in RULE_LEARNERS for m in models)
    ax.axvline(n_rule - 0.5, color=SURFACE, linewidth=6)
    label = metric.replace("_", " ").replace("roc auc", "ROC AUC")
    n = int(summary["n_runs"].max()) if "n_runs" in summary else 1
    splits = f"mean over {n} splits" if n > 1 else "one split"
    ax.set_title(f"Test {label}, {splits} (best per row in bold)", pad=28)
    return _save(fig, path)


def accuracy_vs_rules(summary: pd.DataFrame, path: Path) -> Path:
    """One panel per dataset: accuracy against number of rules."""
    _style()
    datasets = _datasets(summary)
    rule_models = [m for m in MODEL_ORDER if MODELS[m].is_rule_model]
    ncols = 4
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), squeeze=False)

    to_label = []
    for ax, ds in zip(axes.flat, datasets, strict=False):
        sub = summary[summary["dataset"] == ds].set_index("model")
        if "random_forest" in sub.index:
            rf = sub.loc["random_forest", "accuracy_mean"]
            ax.axhline(rf, color=INK_3, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        points = []
        for m in rule_models:
            if m not in sub.index or np.isnan(sub.loc[m, "n_rules_mean"]):
                continue
            x, y = max(sub.loc[m, "n_rules_mean"], 1), sub.loc[m, "accuracy_mean"]
            color = SERIES[0] if m in RULE_LEARNERS else SERIES[1]
            ax.scatter(x, y, s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            points.append((x, y, m))

        ys = [p[1] for p in points]
        top = max(ys + [sub["accuracy_mean"].max()]) if ys else 1.0
        lo_y = min(ys) - 0.06 if ys else 0.0
        hi_y = min(1.0, top + 0.04)
        ax.set_ylim(lo_y, hi_y)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 1500)
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(ds, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        to_label.append((ax, points))

    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    for ax in axes[-1, :]:
        ax.set_xlabel("number of rules (log scale)")

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=SERIES[0], label="optimisation-based rule learner"
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", color=SERIES[1], label="decision tree (one rule per leaf)"
        ),
        plt.Line2D([], [], color=INK_3, linestyle=(0, (4, 3)), label="Random Forest accuracy"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.02))
    fig.suptitle(
        "Accuracy against model size: up and to the left is better",
        x=0.0,
        y=1.07,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    for ax, points in to_label:
        _label_points(ax, points)
    return _save(fig, path)


def _label_points(ax, points) -> None:
    """Direct labels next to each point, moved up or down until they do not overlap.

    Works in display coordinates, so call it after the axis limits are final.
    """
    fig = ax.figure
    fig.canvas.draw()  # make transforms final
    scale = fig.dpi / 72
    char_w, line_h = 4.6 * scale, 10 * scale
    placed: list[tuple[float, float, float, float]] = []
    r = 4 * scale  # markers are obstacles too
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        placed.append((px - r, py - r, px + r, py + r))
    # Right of the point first, then left, then a little above/below.
    candidates = [
        (6, 0, "left"),
        (-6, 0, "right"),
        (6, 10, "left"),
        (6, -10, "left"),
        (-6, 10, "right"),
        (-6, -10, "right"),
        (6, 20, "left"),
        (6, -20, "left"),
    ]
    for x, y, m in sorted(points, key=lambda p: -p[1]):
        px, py = ax.transData.transform((x, y))
        w = char_w * len(NAMES[m])
        for dx, dy, ha in candidates:
            x0 = px + dx * scale if ha == "left" else px + dx * scale - w
            y0 = py + dy * scale - line_h / 2
            box = (x0, y0, x0 + w, y0 + line_h)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                break
        placed.append(box)
        ax.annotate(
            NAMES[m],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            ha=ha,
            va="center",
            zorder=2,
            bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none", "alpha": 0.85},
        )


def rule_set_shape(summary: pd.DataFrame, path: Path) -> Path:
    """Grouped bars: number of rules and average rule width per rule learner."""
    _style()
    models = [m for m in RULE_LEARNERS if m in set(summary["model"])]
    datasets = _datasets(summary)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    width = 0.8 / len(models)
    x = np.arange(len(datasets))

    panels = [
        ("n_rules_mean", "Number of rules (log scale; 0 = only a default rule)"),
        ("avg_rule_width_mean", "Average rule width (features per rule)"),
    ]
    for ax, (col, title) in zip(axes, panels, strict=True):
        for k, m in enumerate(models):
            vals = (
                summary[summary["model"] == m]
                .set_index("dataset")[col]
                .reindex(datasets)
                .to_numpy(dtype=float)
            )
            pos = x - 0.4 + width * (k + 0.5)
            ax.bar(
                pos,
                np.nan_to_num(vals),
                width=width - 0.03,
                color=SERIES[k],
                label=NAMES[m],
                edgecolor=SURFACE,
                linewidth=0.8,
            )
            base = 0.42 if col == "n_rules_mean" else 0
            for p, v in zip(pos, vals, strict=True):
                if np.isnan(v):
                    ax.text(p, base, "–", ha="center", va="bottom", fontsize=8, color=INK_3)
                elif v == 0 and col == "n_rules_mean":
                    ax.text(p, base, "0", ha="center", va="bottom", fontsize=7.5, color=INK_3)
        if col == "n_rules_mean":
            # Log scale: RuleOpt's ~100 rules would otherwise flatten CORELS's single rule.
            ax.set_yscale("log")
            ax.set_ylim(0.4, None)
            ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[1].set_xticks(x, datasets)
    axes[0].legend(ncol=len(models), loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.text(
        0.0,
        -0.01,
        "– = not run (CORELS: binary targets only; IDS: more than 5000 training rows).",
        fontsize=8,
        color=INK_3,
    )
    fig.tight_layout()
    return _save(fig, path)


def thesis_vs_reproduced(summary: pd.DataFrame, thesis: pd.DataFrame, path: Path) -> Path:
    """Accuracy from Table 2 of the thesis against the reproduced mean."""
    _style()
    acc = (
        thesis[thesis["metric"] == "accuracy"]
        .melt(id_vars=["metric", "dataset"], var_name="model", value_name="thesis")
        .dropna()
    )
    now = summary[["dataset", "model", "accuracy_mean", "accuracy_std"]]
    df = acc.merge(now, on=["dataset", "model"], how="inner")

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.plot([0.3, 1.0], [0.3, 1.0], color=INK_3, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.errorbar(
        df["thesis"],
        df["accuracy_mean"],
        yerr=df["accuracy_std"],
        fmt="none",
        ecolor=BLUE_RAMP[2],
        elinewidth=1,
        zorder=2,
    )
    ax.scatter(
        df["thesis"], df["accuracy_mean"], s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3
    )
    for _, r in df.iterrows():
        if abs(r["accuracy_mean"] - r["thesis"]) >= 0.08:
            right = r["thesis"] > 0.75
            ax.annotate(
                f"{NAMES[r['model']]}, {r['dataset']}",
                (r["thesis"], r["accuracy_mean"]),
                xytext=(-7 if right else 7, 0),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
                ha="right" if right else "left",
                va="center",
            )
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("accuracy in the thesis (Table 2, one split)")
    ax.set_ylabel("accuracy now (mean ± sd over splits)")
    diff = (df["accuracy_mean"] - df["thesis"]).abs()
    ax.set_title(f"Thesis vs. this repository\nmedian absolute difference {diff.median():.3f}")
    return _save(fig, path)
