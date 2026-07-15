"""Pydantic v2 configuration and result models for the WOE credit scoring library."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PipelineConfig(BaseModel):
    """Configuration for the WOE credit scoring pipeline."""

    iv_threshold: float = Field(default=0.02, ge=0, description="Minimum IV value for feature selection.")
    normalization_threshold: float = Field(default=0.05, ge=0, le=1, description="Threshold for discrete category normalization.")
    pdo: int = Field(default=20, gt=0, description="Points to double the odds.")
    base_score: int = Field(default=600, ge=300, le=900, description="Base score corresponding to base odds.")
    base_odds: float = Field(default=50, gt=0, description="Odds ratio at the base score.")
    min_score: int = Field(default=400, ge=0, description="Minimum allowable score.")
    max_score: int = Field(default=900, gt=0, description="Maximum allowable score.")
    discretization_method: str = Field(default="quantile", description="Method used for continuous feature discretization.")
    max_discretization_bins: int = Field(default=6, ge=2, le=20, description="Maximum number of bins for discretization.")
    strictly_monotonic: bool = Field(default=True, description="Enforce strictly monotonic WoE bins.")
    n_threads: int = Field(default=1, ge=1, le=64, description="Number of threads for parallel execution.")
    treat_outliers: bool = Field(default=False, description="Apply outlier treatment before discretization.")
    outlier_threshold: float = Field(default=0.01, ge=0, le=0.5, description="Percentile threshold for outlier detection.")

    @model_validator(mode="after")
    def _validate_score_range(self) -> "PipelineConfig":
        if self.min_score >= self.max_score:
            raise ValueError(f"min_score ({self.min_score}) must be strictly less than max_score ({self.max_score}).")
        return self


class FeatureInfo(BaseModel):
    """Summary information for a single feature after WOE scoring."""

    feature: str = Field(..., description="Feature name.")
    iv: float = Field(..., ge=0, description="Information Value of the feature.")
    feature_type: str = Field(default="unknown", description="Feature type: 'continuous' or 'discrete'.")
    status: Literal["selected", "rejected"] = Field(default="selected", description="Selection status after IV screening.")


class ScorecardResult(BaseModel):
    """Complete result of a scorecard build, including performance metrics."""

    features: list[FeatureInfo] = Field(..., description="Per-feature IV and selection info.")
    auc_train: float = Field(..., description="Area under the ROC curve on training set.")
    auc_valid: float = Field(..., description="Area under the ROC curve on validation set.")
    n_features_total: int = Field(..., ge=0, description="Total number of features evaluated.")
    n_features_selected: int = Field(..., ge=0, description="Number of features selected for the scorecard.")
    overfitting_warning: bool = Field(default=False, description="True if a significant gap exists between train and validation AUC.")
    score_range: tuple[float, float] = Field(..., description="Min and max score observed (min, max).")
    created_at: str = Field(default="", description="ISO-8601 datetime of scorecard creation.")


class DatasetProfile(BaseModel):
    """Descriptive profile of the dataset used to build the scorecard."""

    n_rows: int = Field(..., ge=0, description="Number of rows in the dataset.")
    n_columns: int = Field(..., ge=0, description="Number of columns in the dataset.")
    n_continuous: int = Field(..., ge=0, description="Number of continuous features.")
    n_discrete: int = Field(..., ge=0, description="Number of discrete features.")
    target_rate: float = Field(..., ge=0, le=1, description="Proportion of positive class in the target variable.")
    missing_pct: dict[str, float] = Field(..., description="Missing value percentage per column.")
    timestamp: str = Field(default="", description="ISO-8601 datetime of profiling.")
