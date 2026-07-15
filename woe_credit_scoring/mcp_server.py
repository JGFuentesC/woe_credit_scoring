"""
MCP server exposing WOE Credit Scoring toolkit as LLM-callable tools.
"""

import json
import pickle
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from woe_credit_scoring import AutoCreditScoring, IVCalculator, dataset_profile

_MCP_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    try:
        from fastmcp import FastMCP
        _MCP_AVAILABLE = True
    except ImportError:
        pass

if _MCP_AVAILABLE:
    mcp = FastMCP("WOE Credit Scoring")
else:
    mcp = None


def _detect_feature_types(df: pd.DataFrame, target: str) -> tuple:
    features = [c for c in df.columns if c != target]
    continuous = []
    discrete = []
    for col in features:
        if col.startswith("C_"):
            continuous.append(col)
        elif col.startswith("D_"):
            discrete.append(col)
        elif df[col].dtype == object or df[col].dtype.name == "category":
            discrete.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            continuous.append(col)
        else:
            discrete.append(col)
    return continuous, discrete


def _load_model(model_path: str) -> AutoCreditScoring:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    if not isinstance(model, AutoCreditScoring) or not model.is_fitted:
        raise ValueError("Model is not a fitted AutoCreditScoring instance")
    return model


def _sanitize(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return json.loads(value.to_json(orient="records"))
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _check_mcp():
    if mcp is None:
        print(
            "fastmcp is not installed. Install with: pip install woe_credit_scoring[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)


if _MCP_AVAILABLE:

    @mcp.tool()
    def analyze_dataset(data_path: str, target: str) -> dict:
        """Profile a credit scoring dataset: rows, columns, target distribution, missing values, feature types.

        Args:
            data_path: Path to CSV file.
            target: Target column name.
        """
        _check_mcp()
        df = pd.read_csv(data_path)
        cont, disc = _detect_feature_types(df, target)
        profile = dataset_profile(df, target, discrete_features=disc, continuous_features=cont)
        return _sanitize(profile)

    @mcp.tool()
    def calculate_iv(data_path: str, target: str, max_bins: int = 6, method: str = "quantile") -> dict:
        """Calculate Information Value for all features in a dataset.

        Args:
            data_path: Path to CSV file.
            target: Target column name.
            max_bins: Maximum discretization bins.
            method: Discretization method.
        """
        _check_mcp()
        df = pd.read_csv(data_path)
        cont, disc = _detect_feature_types(df, target)
        calculator = IVCalculator(
            data=df,
            target=target,
            continuous_features=cont,
            discrete_features=disc,
        )
        result = calculator.calculate_iv(
            max_discretization_bins=max_bins,
            discretization_method=method,
        )
        return _sanitize(result)

    @mcp.tool()
    def build_scorecard(
        data_path: str,
        target: str,
        model_path: str = None,
        iv_threshold: float = 0.05,
        max_bins: int = 6,
        method: str = "quantile",
        calibrate: bool = False,
    ) -> dict:
        """Build a WOE credit scorecard and optionally save the model.

        Args:
            data_path: Path to CSV file.
            target: Target column name.
            model_path: Optional path to save the fitted model (.pkl).
            iv_threshold: IV threshold for feature selection.
            max_bins: Maximum discretization bins.
            method: Discretization method.
            calibrate: Whether to calibrate probabilities.
        """
        _check_mcp()
        df = pd.read_csv(data_path)
        cont, disc = _detect_feature_types(df, target)
        if not cont and not disc:
            raise ValueError("No features detected. Use C_ prefix for continuous, D_ prefix for discrete.")

        model = AutoCreditScoring(
            data=df,
            target=target,
            continuous_features=cont,
            discrete_features=disc,
        )
        model.fit(
            max_discretization_bins=max_bins,
            iv_feature_threshold=iv_threshold,
            discretization_method=method,
            calibrate=calibrate,
        )

        if model_path:
            model.to_pickle(model_path)

        feature_analysis = _sanitize(model.feature_analysis())
        iv_report = _sanitize(model.iv_report) if hasattr(model, "iv_report") else []

        scores = model.scored_train.copy()
        score_summary = {
            "train": {
                "min": scores["score"].min(),
                "max": scores["score"].max(),
                "mean": round(float(scores["score"].mean()), 2),
            },
            "auc_train": round(float(model.auc_train), 4),
            "auc_valid": round(float(model.auc_valid), 4),
        }

        return {
            "status": "fitted",
            "candidate_features": model.candidate_features,
            "score_summary": score_summary,
            "feature_analysis": feature_analysis,
            "iv_report": iv_report,
            "model_path": model_path,
        }

    @mcp.tool()
    def score_clients(model_path: str, data_path: str) -> list[dict]:
        """Score new clients using a saved model.

        Args:
            model_path: Path to the saved model (.pkl).
            data_path: Path to CSV with client data.
        """
        _check_mcp()
        model = _load_model(model_path)
        df = pd.read_csv(data_path)
        scores = model.predict(df)
        return _sanitize(scores)

    @mcp.tool()
    def explain_decision(model_path: str, client_data: dict) -> list[dict]:
        """Explain the score breakdown for a single client.

        Args:
            model_path: Path to the saved model (.pkl).
            client_data: Dict of feature_name: value for one client.
        """
        _check_mcp()
        model = _load_model(model_path)
        explanation = model.explain(client_data)
        return _sanitize(explanation)


def main():
    _check_mcp()
    mcp.run()


if __name__ == "__main__":
    main()
