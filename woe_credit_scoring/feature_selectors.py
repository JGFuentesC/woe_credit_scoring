from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging
from .base import WoeBaseFeatureSelector
from .discretizer import Discretizer

logger = logging.getLogger("CreditScoringToolkit")


class WoeContinuousFeatureSelector(WoeBaseFeatureSelector):
    """
    Selects continuous features based on Weight of Evidence (WoE) and
    Information Value (IV). Supports multiple discretization strategies
    including 'quantile', 'uniform', 'kmeans', 'gaussian', 'dcc', and 'dec'.
    Can enforce monotonic risk behavior for selected features.

    Attributes:
        selected_features: List of selected features with their IV values.
        iv_report: DataFrame containing the IV report for all features.
    """

    selected_features: Optional[List[Dict[str, float]]] = None
    __is_fitted: bool = False
    _Xd: Optional[pd.DataFrame] = None
    discretizers: Optional[List[Discretizer]] = None
    iv_report: Optional[pd.DataFrame] = None

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        status = "fitted" if self.__is_fitted else "not fitted"
        n_feat = len(self.selected_features) if self.selected_features else 0
        return f"WoeContinuousFeatureSelector({status}, {n_feat} features selected)"

    def get_params(self, deep: bool = True) -> Dict:
        return {}

    def set_params(self, **params) -> 'WoeContinuousFeatureSelector':
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = 'quantile',
            iv_threshold: float = 0.1, min_bins: int = 2, max_bins: int = 5,
            n_threads: int = 1, strictly_monotonic: bool = False) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')
        if not isinstance(y, pd.Series):
            raise TypeError('y must be a pandas Series')

        cont_features = list(X.columns)
        methods = ['quantile', 'uniform', 'kmeans', 'gaussian']

        if method not in methods + ['dcc', 'dec']:
            raise Exception('Invalid method, options are quantile, uniform, kmeans, gaussian, dcc and dec')

        if method in methods:
            discretizers = [Discretizer(strategy=method, min_segments=min_bins, max_segments=max_bins)]
        else:
            discretizers = [Discretizer(strategy=m, min_segments=min_bins, max_segments=max_bins) for m in methods]

        for disc in discretizers:
            disc.fit(X[cont_features], n_threads=n_threads)

        self.discretizers = discretizers
        self._Xd = pd.concat([disc.transform(X[cont_features]) for disc in discretizers], axis=1)
        disc_features = list(self._Xd.columns)
        self._Xd['binary_target'] = y

        if strictly_monotonic:
            mono = {feature: self._check_monotonic(self._Xd[feature], self._Xd['binary_target'])
                    for feature in disc_features}
            mono = {x: y_val for x, y_val in mono.items() if y_val}
            if not mono:
                raise Exception(
                    'There is no monotonic feature. Please try turning strictly_monotonic '
                    'parameter to False or increase the number of bins')
            disc_features = list(mono.keys())

        iv = [(feature, self._information_value(self._Xd[feature], self._Xd['binary_target']))
              for feature in disc_features]
        self.iv_report = pd.DataFrame(iv, columns=['feature', 'iv']).dropna().reset_index(drop=True)
        self.iv_report['relevant'] = self.iv_report['iv'] >= iv_threshold

        self.iv_report['root_feature'] = self.iv_report['feature'].apply(
            lambda x: "_".join(x.split('_')[1:-2]))
        self.iv_report['nbins'] = self.iv_report['feature'].apply(lambda x: x.split('_')[-2])
        self.iv_report['method'] = self.iv_report['feature'].apply(lambda x: x.split('_')[-1])

        sort_columns = (['root_feature', 'iv', 'nbins'] if method in methods + ['dcc']
                        else ['root_feature', 'method', 'iv', 'nbins'])
        self.iv_report = self.iv_report.sort_values(
            by=sort_columns,
            ascending=[True, False, True] if method in methods + ['dcc']
            else [True, True, False, True]
        ).reset_index(drop=True)
        self.iv_report['index'] = (
            self.iv_report.groupby('root_feature').cumcount() + 1
            if method in methods + ['dcc']
            else self.iv_report.groupby(['root_feature', 'method']).cumcount() + 1
        )

        self.iv_report = self.iv_report.loc[self.iv_report['index'] == 1].reset_index(drop=True)
        self.iv_report['selected'] = self.iv_report['feature'].isin(self.iv_report['feature'])
        self.iv_report = self.iv_report.sort_values(
            by=['selected', 'relevant'], ascending=[False, False])
        cont_features_selected = list(set(self.iv_report.loc[self.iv_report['relevant']]['root_feature']))
        if len(cont_features_selected) == 0:
            raise Exception(
                'No relevant feature found. Please try increasing the number of bins '
                'or changing the discretization method')
        for disc in self.discretizers:
            disc.fit(X[cont_features_selected], n_threads=n_threads)
        self.selected_features = (
            self.iv_report[self.iv_report['relevant']]
            .drop('index', axis=1).to_dict(orient='records')
        )
        self.__is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')

        features = list(set([feature['root_feature'] for feature in self.selected_features]))
        non_present_features = [f for f in features if f not in X.columns]

        if non_present_features:
            logger.exception(
                f'{", ".join(non_present_features)} feature'
                f'{"s" if len(non_present_features) > 1 else ""} not present in data')
            raise Exception("Missing features")

        aux = pd.concat([disc.transform(X[features]) for disc in self.discretizers], axis=1)
        aux = aux[[feature['feature'] for feature in self.selected_features]]
        return aux


class WoeDiscreteFeatureSelector(WoeBaseFeatureSelector):
    """
    Selects discrete features based on Weight of Evidence (WoE) and
    Information Value (IV). Evaluates each feature's predictive power
    by calculating its IV and selects features meeting a specified threshold.

    Attributes:
        iv_report: DataFrame with IV values and selection status per feature.
        selected_features: Dict of selected features and their IV values.
    """

    iv_report: pd.DataFrame = None

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        status = "fitted" if self.__is_fitted else "not fitted"
        n_feat = len(self.selected_features) if hasattr(self, 'selected_features') and self.selected_features else 0
        return f"WoeDiscreteFeatureSelector({status}, {n_feat} features selected)"

    def get_params(self, deep: bool = True) -> Dict:
        return {}

    def set_params(self, **params) -> 'WoeDiscreteFeatureSelector':
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, iv_threshold: float = 0.1) -> None:
        disc_features: list = list(X.columns)
        aux: pd.DataFrame = X.copy()
        aux['binary_target'] = y
        iv: list = [(feature, self._information_value(aux[feature], aux['binary_target']))
                     for feature in disc_features]
        self.iv_report = pd.DataFrame(iv, columns=['feature', 'iv']).dropna().reset_index(drop=True)
        self.iv_report['selected'] = self.iv_report['iv'] >= iv_threshold
        self.iv_report = self.iv_report.sort_values('selected', ascending=False)
        disc_features = list(self.iv_report.loc[self.iv_report['selected']]['feature'])
        if len(disc_features) == 0:
            raise Exception('No relevant feature found. Please try decreasing the IV threshold')
        self.selected_features: dict = (
            self.iv_report.loc[self.iv_report['selected']]
            .set_index('feature')['iv'].to_dict()
        )
        self.__is_fitted: bool = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux: pd.DataFrame = X.copy()
            features: list = [feature for feature in self.selected_features.keys()]
            non_present_features: list = [f for f in features if f not in X.columns]
            if len(non_present_features) > 0:
                logger.exception(
                    f'{",".join(non_present_features)} feature'
                    f'{"s" if len(non_present_features) > 1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                aux = aux[features]
                return aux
