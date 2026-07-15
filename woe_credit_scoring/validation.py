from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from .plots import COLORS, _style_axis


class ValidationReport:
    def __init__(
        self,
        y_true_train: np.ndarray,
        y_score_train: np.ndarray,
        y_true_valid: np.ndarray,
        y_score_valid: np.ndarray,
        scores_train: pd.DataFrame,
        scores_valid: pd.DataFrame,
        psi_data: Optional[Dict[str, float]] = None,
    ):
        self._y_true_train = y_true_train
        self._y_score_train = y_score_train
        self._y_true_valid = y_true_valid
        self._y_score_valid = y_score_valid
        self._scores_train = scores_train
        self._scores_valid = scores_valid
        self._psi_data = psi_data

        self._ks_cache: Optional[float] = None
        self._gini_cache: Optional[float] = None
        self._auc_train_cache: Optional[float] = None
        self._auc_valid_cache: Optional[float] = None
        self._overfitting_gap_cache: Optional[float] = None
        self._confusion_matrix_cache: Optional[pd.DataFrame] = None
        self._psi_report_cache: Optional[pd.DataFrame] = None
        self._optimal_threshold_cache: Optional[float] = None

    @property
    def ks(self) -> float:
        if self._ks_cache is None:
            fpr, tpr, _ = roc_curve(self._y_true_train, self._y_score_train)
            ks_train = np.max(tpr - fpr)
            fpr, tpr, _ = roc_curve(self._y_true_valid, self._y_score_valid)
            ks_valid = np.max(tpr - fpr)
            self._ks_cache = max(ks_train, ks_valid)
        return self._ks_cache

    @property
    def gini(self) -> float:
        if self._gini_cache is None:
            self._gini_cache = 2 * self.auc_valid - 1
        return self._gini_cache

    @property
    def auc_train(self) -> float:
        if self._auc_train_cache is None:
            self._auc_train_cache = roc_auc_score(self._y_true_train, self._y_score_train)
        return self._auc_train_cache

    @property
    def auc_valid(self) -> float:
        if self._auc_valid_cache is None:
            self._auc_valid_cache = roc_auc_score(self._y_true_valid, self._y_score_valid)
        return self._auc_valid_cache

    @property
    def overfitting_gap(self) -> float:
        if self._overfitting_gap_cache is None:
            self._overfitting_gap_cache = abs(self.auc_train - self.auc_valid)
        return self._overfitting_gap_cache

    @property
    def _optimal_threshold(self) -> float:
        if self._optimal_threshold_cache is None:
            fpr, tpr, thresholds = roc_curve(self._y_true_valid, self._y_score_valid)
            youden = tpr - fpr
            self._optimal_threshold_cache = thresholds[np.argmax(youden)]
        return self._optimal_threshold_cache

    @property
    def confusion_matrix(self) -> pd.DataFrame:
        if self._confusion_matrix_cache is None:
            y_pred = (self._y_score_valid >= self._optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self._y_true_valid, y_pred).ravel()
            self._confusion_matrix_cache = pd.DataFrame(
                [["TP", tp], ["TN", tn], ["FP", fp], ["FN", fn]],
                columns=["Metric", "Count"],
            )
        return self._confusion_matrix_cache

    @property
    def psi_report(self) -> pd.DataFrame:
        if self._psi_report_cache is None:
            if self._psi_data is None:
                self._psi_report_cache = pd.DataFrame(
                    columns=["feature", "psi", "interpretation"]
                )
            else:
                records = []
                for feature, psi_val in self._psi_data.items():
                    if psi_val < 0.1:
                        interp = "stable"
                    elif psi_val < 0.25:
                        interp = "minor change"
                    else:
                        interp = "significant change"
                    records.append(
                        {"feature": feature, "psi": psi_val, "interpretation": interp}
                    )
                self._psi_report_cache = pd.DataFrame(records)
        return self._psi_report_cache

    def plot_cumulative_gains(self) -> plt.Figure:
        fpr, tpr, _ = roc_curve(self._y_true_valid, self._y_score_valid)
        n = len(self._y_true_valid)
        y_count = np.sum(self._y_true_valid)
        x_gain = fpr * (1 - y_count / n) + tpr * (y_count / n)
        y_gain = tpr

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(x_gain, y_gain, lw=2.2, color=COLORS["accent"], label="Model")
        ax.fill_between(x_gain, y_gain, alpha=0.06, color=COLORS["accent"])
        ax.plot([0, 1], [0, 1], lw=1.2, color=COLORS["diagonal"], linestyle="--", label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Cumulative Population")
        ax.set_ylabel("Cumulative Gains")
        ax.set_title("Cumulative Gains Chart — Validation")
        ax.legend(loc="lower right", frameon=True)
        _style_axis(ax)
        fig.tight_layout(pad=1.2)
        return fig

    def to_dict(self) -> dict:
        threshold = self._optimal_threshold
        cm = self.confusion_matrix.set_index("Metric")["Count"].to_dict()
        result = {
            "ks": self.ks,
            "gini": self.gini,
            "auc_train": self.auc_train,
            "auc_valid": self.auc_valid,
            "overfitting_gap": self.overfitting_gap,
            "optimal_threshold": threshold,
            "true_positives": cm.get("TP", 0),
            "true_negatives": cm.get("TN", 0),
            "false_positives": cm.get("FP", 0),
            "false_negatives": cm.get("FN", 0),
        }
        if self._psi_data is not None:
            result["psi"] = self._psi_data
        return result

    def summarize(self) -> str:
        lines = [
            "=== Validation Report ===",
            f"KS statistic : {self.ks:.4f}",
            f"Gini         : {self.gini:.4f}",
            f"AUC train    : {self.auc_train:.4f}",
            f"AUC valid    : {self.auc_valid:.4f}",
            f"Overfit gap  : {self.overfitting_gap:.4f}",
        ]
        cm = self.confusion_matrix.set_index("Metric")["Count"].to_dict()
        lines.append(f"Opt threshold: {self._optimal_threshold:.4f}")
        lines.append(f"TP: {cm.get('TP', 0)}, TN: {cm.get('TN', 0)}, "
                      f"FP: {cm.get('FP', 0)}, FN: {cm.get('FN', 0)}")
        if self._psi_data is not None:
            lines.append("\n-- PSI Report --")
            for feature, psi_val in sorted(self._psi_data.items(), key=lambda x: -x[1]):
                if psi_val < 0.1:
                    interp = "stable"
                elif psi_val < 0.25:
                    interp = "minor change"
                else:
                    interp = "significant change"
                lines.append(f"  {feature}: {psi_val:.4f} ({interp})")
        return "\n".join(lines)
