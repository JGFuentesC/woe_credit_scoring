from typing import Union
import numpy as np
import pandas as pd

class WoeBaseFeatureSelector:
    """
    Base class for selecting features based on their Weight of Evidence (WoE)
    transformation and Information Value (IV) statistic.

    This class provides foundational methods for evaluating and selecting
    features by transforming them using WoE and calculating their IV.
    The IV statistic is used to measure the predictive power of each feature
    with respect to a binary target variable. Features with higher IV values
    are considered more predictive.

    The class includes methods to compute the IV statistic, check for
    monotonic risk behavior, and other utility functions that can be extended
    by subclasses to implement specific feature selection strategies.

    Attributes:
        None

    Methods:
        _information_value(X, y): Computes the IV statistic for a given feature.
        _check_monotonic(X, y): Checks if a feature exhibits monotonic risk behavior.
    """

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "WoeBaseFeatureSelector()"

    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params) -> 'WoeBaseFeatureSelector':
        return self

    @staticmethod
    def _information_value(X: pd.Series, y: pd.Series) -> Union[float, None]:
        """
        Computes information value (IV) statistic.

        Args:
            X (pd.Series): Discretized predictors data.
            y (pd.Series): Dichotomic response feature.

        Returns:
            Union[float, None]: IV statistic or None if IV is infinite.

        Reference:
            For more details on the Information Value statistic, see
            http://arxiv.org/pdf/2309.13183
        """
        aux = pd.concat([X, y], axis=1)
        aux.columns = ['x', 'y']
        aux = aux.assign(nrow=1)
        aux = aux.pivot_table(index='x', columns='y',
                              values='nrow', aggfunc='sum', fill_value=0)
        aux /= aux.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            aux['woe'] = np.log(aux.iloc[:, 0] / aux.iloc[:, 1])
            aux['iv'] = (aux.iloc[:, 0] - aux.iloc[:, 1]) * aux['woe']
        iv = aux['iv'].sum()
        if np.isinf(iv) or np.isnan(iv):
            return None
        return iv

    @staticmethod
    def _check_monotonic(X: pd.Series, y: pd.Series) -> bool:
        """
        Validates if a given discretized feature has monotonic risk behavior.

        Args:
            X (pd.Series): Discretized predictors data.
            y (pd.Series): Dichotomic response feature.

        Returns:
            bool: Whether or not the feature has monotonic risk.
        """
        aux = pd.concat([X, y], axis=1)
        aux.columns = ['x', 'y']
        aux = aux.loc[aux['x'] != 'MISSING'].reset_index(drop=True)
        aux = aux.groupby('x').mean()
        y_values = list(aux['y'])
        return (len(y_values) >= 2) and (sorted(y_values) == y_values or sorted(y_values, reverse=True) == y_values)
