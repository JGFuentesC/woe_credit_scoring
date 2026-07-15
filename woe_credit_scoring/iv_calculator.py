from typing import List
import numpy as np
import pandas as pd
import logging
from .feature_selectors import WoeContinuousFeatureSelector, WoeDiscreteFeatureSelector
from .normalizer import DiscreteNormalizer

logger = logging.getLogger("CreditScoringToolkit")


class IVCalculator:
    """
    Calculates the Information Value (IV) for both discrete and continuous features.

    Provides a simple interface that abstracts away the manual steps of
    discretization and normalization.

    Example:
        >>> from woe_credit_scoring import IVCalculator
        >>> import pandas as pd
        >>> data = pd.read_csv('example_data/hmeq.csv')
        >>> iv_calculator = IVCalculator(
        ...     data=data, target='BAD',
        ...     continuous_features=['LOAN', 'MORTDUE'],
        ...     discrete_features=['REASON', 'JOB']
        ... )
        >>> iv_report = iv_calculator.calculate_iv()
        >>> print(iv_report)
    """

    def __init__(self, data: pd.DataFrame, target: str,
                 continuous_features: List[str] = None,
                 discrete_features: List[str] = None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in the DataFrame.")

        self.data = data
        self.target = target
        self.continuous_features = continuous_features if continuous_features is not None else []
        self.discrete_features = discrete_features if discrete_features is not None else []

        if not self.continuous_features and not self.discrete_features:
            logger.warning("No continuous or discrete features provided.")

    def __repr__(self) -> str:
        return (f"IVCalculator(target='{self.target}', "
                f"continuous={len(self.continuous_features)}, "
                f"discrete={len(self.discrete_features)})")

    def get_params(self, deep: bool = True) -> dict:
        return {
            'target': self.target,
            'continuous_features': self.continuous_features,
            'discrete_features': self.discrete_features,
        }

    def set_params(self, **params) -> 'IVCalculator':
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def calculate_iv(self,
                     max_discretization_bins: int = 5,
                     strictly_monotonic: bool = False,
                     discretization_method: str = 'quantile',
                     n_threads: int = 1,
                     discrete_normalization_threshold: float = 0.05,
                     discrete_normalization_default_category: str = 'OTHER'
                     ) -> pd.DataFrame:
        iv_reports = []

        if self.continuous_features:
            logger.info("Calculating IV for continuous features...")
            woe_continuous_selector = WoeContinuousFeatureSelector()
            try:
                woe_continuous_selector.fit(
                    self.data[self.continuous_features],
                    self.data[self.target],
                    max_bins=max_discretization_bins,
                    strictly_monotonic=strictly_monotonic,
                    iv_threshold=-np.inf,
                    method=discretization_method,
                    n_threads=n_threads
                )
                iv_report_continuous = woe_continuous_selector.iv_report
                iv_report_continuous = (
                    iv_report_continuous[['root_feature', 'iv']]
                    .rename(columns={'root_feature': 'feature'})
                )
                iv_report_continuous['feature_type'] = 'continuous'
                iv_reports.append(iv_report_continuous)
                logger.info("IV for continuous features calculated successfully.")
            except Exception as e:
                logger.error(f"Could not calculate IV for continuous features. Error: {e}")

        if self.discrete_features:
            logger.info("Calculating IV for discrete features...")
            try:
                dn = DiscreteNormalizer(
                    normalization_threshold=discrete_normalization_threshold,
                    default_category=discrete_normalization_default_category
                )
                dn.fit(self.data[self.discrete_features])
                normalized_discrete_data = dn.transform(self.data[self.discrete_features])

                woe_discrete_selector = WoeDiscreteFeatureSelector()
                woe_discrete_selector.fit(
                    normalized_discrete_data,
                    self.data[self.target],
                    iv_threshold=-np.inf
                )
                iv_report_discrete = woe_discrete_selector.iv_report[['feature', 'iv']]
                iv_report_discrete['feature_type'] = 'discrete'
                iv_reports.append(iv_report_discrete)
                logger.info("IV for discrete features calculated successfully.")
            except Exception as e:
                logger.error(f"Could not calculate IV for discrete features. Error: {e}")

        if not iv_reports:
            logger.warning("IV calculation did not produce any results.")
            return pd.DataFrame(columns=['feature', 'iv', 'feature_type'])

        final_iv_report = pd.concat(iv_reports, axis=0).sort_values(
            'iv', ascending=False).reset_index(drop=True)
        return final_iv_report
