"""Visualization utilities for WOE credit scoring.

Standalone plotting functions that return matplotlib Figure objects.
All functions accept an optional `ax` parameter to support subplot composition.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def roc_curve_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    label: str = "Model",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Plot ROC curve with AUC annotation.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def roc_comparison_plot(
    y_true_train: pd.Series,
    y_score_train: np.ndarray,
    y_true_valid: pd.Series,
    y_score_valid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Plot ROC curves for train and validation sets side by side.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for y_true, scores, name, color in [
        (y_true_train, y_score_train, "Train", "#3498db"),
        (y_true_valid, y_score_valid, "Valid", "#e74c3c"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Train vs Validation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def ks_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot Kolmogorov-Smirnov (KS) chart with cumulative distributions.

    Shows cumulative % of goods and bads across score deciles,
    with the KS statistic annotated at the point of maximum separation.

    Returns:
        matplotlib.figure.Figure, float (KS statistic)
    """
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

    x = all_deciles
    ax.plot(x, cum_goods.values, "g-", lw=2, label="Cumulative Goods")
    ax.plot(x, cum_bads.values, "r-", lw=2, label="Cumulative Bads")
    ax.vlines(ks_decile, cum_goods.iloc[ks_decile], cum_bads.iloc[ks_decile],
              colors="k", linestyles="--", lw=1.5)

    ax.annotate(
        f"KS = {ks_stat:.4f}",
        xy=(ks_decile, (cum_goods.iloc[ks_decile] + cum_bads.iloc[ks_decile]) / 2),
        xytext=(ks_decile + 0.5, (cum_goods.iloc[ks_decile] + cum_bads.iloc[ks_decile]) / 2 + 0.05),
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )

    ax.set_xlabel("Score Decile")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("KS Chart")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def iv_barplot(
    iv_report: pd.DataFrame,
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Horizontal barplot of Information Value by feature.

    Args:
        iv_report: DataFrame with columns 'feature' and 'iv'.
        top_n: Number of top features to display.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data = (
        iv_report.sort_values("iv", ascending=True)
        .tail(top_n)
        .reset_index(drop=True)
    )
    colors = ["#2ecc71" if v >= 0.1 else "#f39c12" if v >= 0.02 else "#e74c3c"
              for v in data["iv"]]

    ax.barh(data["feature"], data["iv"], color=colors)
    ax.axvline(0.1, color="green", linestyle="--", alpha=0.5, label="IV = 0.1 (medium)")
    ax.axvline(0.02, color="orange", linestyle="--", alpha=0.5, label="IV = 0.02 (weak)")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title(f"Top {top_n} Features by Information Value")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def event_rate_plot(
    df: pd.DataFrame,
    target: str,
    feature: str,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Barplot of event rate by category for a given discrete feature.

    Args:
        df: DataFrame with feature and target columns.
        target: Target column name (binary).
        feature: Feature column name to analyze.

    Returns:
        matplotlib.figure.Figure
    """
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

    ax.barh(event_rate.index, event_rate["event_rate"], color="#3498db")
    ax.set_xlabel("Event Rate")
    ax.set_ylabel(feature)
    ax.set_title(f"Event Rate by {feature}")

    for i, (rate, n) in enumerate(zip(event_rate["event_rate"], event_rate["n"])):
        ax.text(rate + 0.005, i, f"n={n}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def score_distribution_plot(
    scores_train: pd.Series,
    scores_valid: Optional[pd.Series] = None,
    n_bins: int = 20,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Overlaid histogram of score distributions for train (and optionally valid).

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(scores_train, bins=n_bins, alpha=0.6, label="Train",
            color="#3498db", density=True)
    if scores_valid is not None:
        ax.hist(scores_valid, bins=n_bins, alpha=0.6, label="Valid",
                color="#e74c3c", density=True)

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def event_rate_by_score_plot(
    df: pd.DataFrame,
    target: str,
    score_col: str = "score",
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Stacked bar chart of good/bad proportions by score range.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    df = df.copy()
    df["score_range"] = pd.cut(df[score_col], bins=n_bins, include_lowest=True)
    crosstab = pd.crosstab(df["score_range"], df[target], normalize="index")

    crosstab.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title(f"Event Rate by Score Range ({n_bins} bins)")
    ax.set_xlabel("Score Range")
    ax.set_ylabel("Proportion")
    ax.legend(["Good (0)", "Bad (1)"])
    fig.tight_layout()
    return fig
