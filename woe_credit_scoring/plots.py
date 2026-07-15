"""Visualization utilities for WOE credit scoring.

Clean, minimal plotting functions with a violet + blue color palette.
All functions return matplotlib Figure objects and accept an optional `ax` parameter.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc


COLORS = {
    "train": "#8B5CF6",
    "valid": "#7DD3FC",
    "good": "#A78BFA",
    "bad": "#F43F5E",
    "bg": "#FAFAFA",
    "grid": "#E5E7EB",
    "text": "#1F2937",
    "accent": "#6D28D9",
    "diagonal": "#D1D5DB",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": COLORS["grid"],
    "axes.facecolor": COLORS["bg"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "legend.edgecolor": COLORS["grid"],
    "legend.framealpha": 0.9,
    "legend.fontsize": 9,
})


def _style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    if grid:
        ax.grid(True, axis="y", alpha=0.6)
        ax.set_axisbelow(True)


def roc_curve_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    label: str = "Model",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6.5, 5.5),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2.2, color=COLORS["accent"], label=f"{label}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1.2, color=COLORS["diagonal"], linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", frameon=True)
    _style_axis(ax)
    fig.tight_layout(pad=1.2)
    return fig


def roc_comparison_plot(
    y_true_train: pd.Series,
    y_score_train: np.ndarray,
    y_true_valid: pd.Series,
    y_score_valid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6.5, 5.5),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for y_true, scores, name, color in [
        (y_true_train, y_score_train, "Train", COLORS["train"]),
        (y_true_valid, y_score_valid, "Valid", COLORS["valid"]),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], lw=1.2, color=COLORS["diagonal"], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", frameon=True)
    _style_axis(ax)
    fig.tight_layout(pad=1.2)
    return fig


def ks_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    label: str = "Validation",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7.5, 5.5),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    df = pd.DataFrame({"target": y_true.values, "score": y_score})
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")

    all_deciles = sorted(df["decile"].unique())
    goods = df[df["target"] == 0].groupby("decile").size()
    bads = df[df["target"] == 1].groupby("decile").size()

    cum_goods = goods.reindex(all_deciles, fill_value=0).cumsum() / goods.sum()
    cum_bads = bads.reindex(all_deciles, fill_value=0).cumsum() / bads.sum()

    ks_values = np.abs(cum_goods - cum_bads)
    ks_stat = ks_values.max()
    ks_decile = ks_values.idxmax()

    ax.plot(all_deciles, cum_goods.values, lw=2.2, color=COLORS["accent"], label="Cumulative Goods")
    ax.fill_between(all_deciles, cum_goods.values, alpha=0.08, color=COLORS["accent"])
    ax.plot(all_deciles, cum_bads.values, lw=2.2, color=COLORS["bad"], label="Cumulative Bads")
    ax.fill_between(all_deciles, cum_bads.values, alpha=0.08, color=COLORS["bad"])

    y_top = cum_goods.iloc[ks_decile]
    y_bot = cum_bads.iloc[ks_decile]
    ax.vlines(ks_decile, y_bot, y_top, colors=COLORS["text"], linestyles="--", lw=1.2)
    ax.scatter([ks_decile], [y_top], color=COLORS["accent"], s=25, zorder=5)
    ax.scatter([ks_decile], [y_bot], color=COLORS["bad"], s=25, zorder=5)

    ax.annotate(
        f"KS = {ks_stat:.4f}",
        xy=(ks_decile, (y_top + y_bot) / 2),
        xytext=(ks_decile + 0.6, (y_top + y_bot) / 2 + 0.08),
        fontsize=11,
        fontweight="600",
        color=COLORS["text"],
        arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["text"]),
    )

    ax.set_xlabel("Score Decile")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(f"KS Chart — {label}")
    ax.legend(loc="best", frameon=True)
    ax.set_ylim(0, 1.02)
    _style_axis(ax, grid=False)
    fig.tight_layout(pad=1.2)
    return fig


def iv_barplot(
    iv_report: pd.DataFrame,
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8.5, 6),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data = (
        iv_report.sort_values("iv", ascending=True)
        .tail(top_n)
        .reset_index(drop=True)
    )
    colors = [
        COLORS["accent"] if v >= 0.1 else "#C4B5FD" if v >= 0.02 else COLORS["bad"]
        for v in data["iv"]
    ]

    bars = ax.barh(data["feature"], data["iv"], color=colors, height=0.65)
    for bar, val in zip(bars, data["iv"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=COLORS["text"])

    ax.axvline(0.1, color="#A78BFA", linestyle="--", lw=1, alpha=0.7, label="IV = 0.1  (medium)")
    ax.axvline(0.02, color="#DDD6FE", linestyle="--", lw=1, alpha=0.7, label="IV = 0.02  (weak)")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title(f"Top {top_n} Features by Information Value")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_xlim(0, data["iv"].max() * 1.25)
    _style_axis(ax)
    fig.tight_layout(pad=1.2)
    return fig


def event_rate_plot(
    df: pd.DataFrame,
    target: str,
    feature: str,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 4.5),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    event_rate = (
        df.groupby(feature)[target]
        .agg(["count", "mean"])
        .rename(columns={"mean": "event_rate", "count": "n"})
        .sort_values("event_rate")
    )

    colors = [COLORS["accent"] if r >= event_rate["event_rate"].mean() else "#C4B5FD"
              for r in event_rate["event_rate"]]

    bars = ax.barh(event_rate.index, event_rate["event_rate"], color=colors, height=0.6)
    ax.axvline(event_rate["event_rate"].mean(), color=COLORS["accent"], linestyle="--",
               lw=1, alpha=0.5, label=f"Mean = {event_rate['event_rate'].mean():.3f}")

    for bar, (_, row) in zip(bars, event_rate.iterrows()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"n={row['n']}", va="center", fontsize=8, color=COLORS["text"])

    ax.set_xlabel("Event Rate")
    ax.set_ylabel(feature)
    ax.set_title(f"Event Rate by {feature}")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    _style_axis(ax)
    fig.tight_layout(pad=1.2)
    return fig


def score_distribution_plot(
    scores_train: pd.Series,
    scores_valid: Optional[pd.Series] = None,
    n_bins: int = 25,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7.5, 5),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(scores_train, bins=n_bins, alpha=0.55, label="Train",
            color=COLORS["train"], density=True, edgecolor="white", linewidth=0.5)
    if scores_valid is not None:
        ax.hist(scores_valid, bins=n_bins, alpha=0.55, label="Valid",
                color=COLORS["valid"], density=True, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend(frameon=True)
    _style_axis(ax)
    fig.tight_layout(pad=1.2)
    return fig


def event_rate_by_score_plot(
    df: pd.DataFrame,
    target: str,
    score_col: str = "score",
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    df = df.copy()
    df["score_range"] = pd.cut(df[score_col], bins=n_bins, include_lowest=True)
    crosstab = pd.crosstab(df["score_range"], df[target], normalize="index")

    crosstab.plot(kind="bar", stacked=True, ax=ax, color=[COLORS["good"], COLORS["bad"]],
                  width=0.8, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Event Rate by Score Range ({n_bins} bins)")
    ax.set_xlabel("Score Range")
    ax.set_ylabel("Proportion")
    ax.legend(["Good", "Bad"], frameon=True)
    _style_axis(ax, grid=False)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(pad=1.2)
    return fig
