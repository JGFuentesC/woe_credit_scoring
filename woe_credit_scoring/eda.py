from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

__all__ = [
    "dataset_profile",
    "psi",
    "event_rate_by_feature",
    "woe_profile",
    "vif",
]

logger = logging.getLogger("CreditScoringToolkit")


def dataset_profile(
    df: pd.DataFrame,
    target: str,
    discrete_features: Optional[List[str]] = None,
    continuous_features: Optional[List[str]] = None,
) -> Dict:
    """
    Returns a comprehensive profile of the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target column.
        discrete_features (List[str], optional): List of discrete feature names.
        continuous_features (List[str], optional): List of continuous feature names.

    Returns:
        dict: Dictionary containing basic_info, missing_report, target_distribution,
              and feature_types.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    rows, cols = df.shape
    memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

    missing = df.isnull().mean().to_frame("missing_pct")
    missing.index.name = "feature"
    missing = missing.reset_index()

    target_counts = df[target].value_counts().sort_index().to_frame("count")
    target_counts["proportion"] = target_counts["count"] / target_counts["count"].sum()

    discrete_features = discrete_features or []
    continuous_features = continuous_features or []

    if not discrete_features and not continuous_features:
        all_features = [c for c in df.columns if c != target]
        for col in all_features:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                discrete_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                continuous_features.append(col)

    return {
        "basic_info": {
            "rows": rows,
            "columns": cols,
            "memory_mb": round(memory, 2),
        },
        "missing_report": missing,
        "target_distribution": target_counts,
        "feature_types": {
            "discrete": len(discrete_features),
            "continuous": len(continuous_features),
        },
    }


def psi(expected: pd.Series, actual: pd.Series, feature: str) -> float:
    """
    Calculates the Population Stability Index between two distributions.

    PSI = sum( (%actual_i - %expected_i) * ln(%actual_i / %expected_i) )

    Args:
        expected (pd.Series): Series of expected distribution values (counts or proportions).
        actual (pd.Series): Series of actual distribution values (counts or proportions).
        feature (str): Feature name for logging purposes.

    Returns:
        float: PSI value. Returns np.inf if a bin is zero in either distribution.
    """
    expected_vals = expected.values.astype(float)
    actual_vals = actual.values.astype(float)

    expected_prop = expected_vals / expected_vals.sum()
    actual_prop = actual_vals / actual_vals.sum()

    epsilon = 0.0001
    expected_prop = np.clip(expected_prop, epsilon, None)
    actual_prop = np.clip(actual_prop, epsilon, None)

    psi_value = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))

    if np.isinf(psi_value) or np.isnan(psi_value):
        logger.warning(f"PSI is infinite or NaN for feature '{feature}'.")
        return np.inf

    return float(psi_value)


def event_rate_by_feature(df: pd.DataFrame, target: str, feature: str) -> pd.DataFrame:
    """
    Computes the event rate for each category of a feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the binary target column.
        feature (str): Name of the feature column.

    Returns:
        pd.DataFrame: DataFrame with columns Category, count, event_count, event_rate,
                      sorted by event_rate descending.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in DataFrame.")

    result = (
        df.groupby(feature)
        .agg(count=(target, "count"), event_count=(target, "sum"))
        .assign(event_rate=lambda x: x["event_count"] / x["count"])
        .reset_index()
        .rename(columns={feature: "Category"})
        .sort_values("event_rate", ascending=False)
        .reset_index(drop=True)
    )

    return result


def woe_profile(df: pd.DataFrame, target: str, feature: str) -> pd.DataFrame:
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) per category.

    WoE = ln(% non-events / % events)
    IV_contribution = (% non-events - % events) * WoE

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the binary target column.
        feature (str): Name of the feature column.

    Returns:
        pd.DataFrame: DataFrame with columns Category, Count, Events, NonEvents,
                      EventRate, WoE, IV.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in DataFrame.")

    grouped = (
        df.groupby(feature)
        .agg(
            Count=(target, "count"),
            Events=(target, "sum"),
        )
        .reset_index()
    )

    grouped["NonEvents"] = grouped["Count"] - grouped["Events"]
    grouped["EventRate"] = grouped["Events"] / grouped["Count"]

    total_events = grouped["Events"].sum()
    total_nonevents = grouped["NonEvents"].sum()

    epsilon = 0.5
    grouped["DistrEvents"] = (grouped["Events"] + epsilon) / (total_events + epsilon)
    grouped["DistrNonEvents"] = (grouped["NonEvents"] + epsilon) / (total_nonevents + epsilon)

    grouped["WoE"] = np.log(grouped["DistrNonEvents"] / grouped["DistrEvents"])
    grouped["IV"] = (grouped["DistrNonEvents"] - grouped["DistrEvents"]) * grouped["WoE"]

    result = grouped.drop(columns=["DistrEvents", "DistrNonEvents"])
    result = result.rename(columns={feature: "Category"})
    result = result[["Category", "Count", "Events", "NonEvents", "EventRate", "WoE", "IV"]]

    return result


def vif(X: pd.DataFrame) -> pd.Series:
    """
    Computes the Variance Inflation Factor (VIF) for each feature.

    VIF = 1 / (1 - R²), where R² comes from regressing each feature
    on all other features.

    Args:
        X (pd.DataFrame): DataFrame of numerical features.

    Returns:
        pd.Series: VIF values indexed by feature name.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    X = X.select_dtypes(include=[np.number]).dropna()

    vif_values = {}
    for i, col in enumerate(X.columns):
        y = X.iloc[:, i].values
        X_other = X.drop(columns=[col]).values
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_other, y)
        r2 = lr.score(X_other, y)
        vif_values[col] = float(np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2))

    return pd.Series(vif_values).sort_values(ascending=False)
