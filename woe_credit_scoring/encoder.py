from typing import Dict
from collections import ChainMap
import numpy as np
import pandas as pd

class WoeEncoder:
    """
    WoeEncoder is a class for encoding discrete features into Weight of Evidence (WoE) values.

    WoE is a commonly used technique in credit scoring and other binary classification problems.
    It transforms categorical features into continuous values based on the log odds of the target variable.

    This class provides methods to fit the WoE transformation based on input data, transform new data using the learned WoE encoding,
    and inverse transform WoE encoded data back to the original categorical values.

    Attributes:
        features (list): List of feature names to be encoded.
        _woe_encoding_map (dict): Dictionary mapping features to their WoE encoding.
        __is_fitted (bool): Flag indicating whether the encoder has been fitted.
        _woe_reverse_map (dict): Dictionary mapping WoE values back to original feature values.

    Reference:
        For more details on the Weight of Evidence (WoE) encoding, see
        http://listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """

    def __init__(self) -> None:
        self.features = None
        self._woe_encoding_map = None
        self.__is_fitted = False
        self._woe_reverse_map = None

    def fit(self, X: pd.DataFrame, y: pd.Series, target_col: str = 'binary_target') -> None:
        """Learns WoE encoding.

        Args:
            X (pd.DataFrame): Data with discrete features.
            y (pd.Series): Dichotomic response.
            target_col (str): Name of the target column to be created in the dataframe.
        """
        aux = X.copy()
        self.features = list(aux.columns)
        aux[target_col] = y
        self._woe_encoding_map = dict(ChainMap(
            *map(lambda feature: self._woe_transformation(aux, feature, target_col), self.features)))
        self.__is_fitted = True

    @staticmethod
    def _woe_transformation(X: pd.DataFrame, feature: str, bin_target: str) -> Dict[str, Dict]:
        """Calculates WoE Map between discrete space and log odds space.

        Args:
            X (pd.DataFrame): Discrete data including dichotomic response feature.
            feature (str): Name of the feature for getting the map.
            bin_target (str): Name of the dichotomic response feature.

        Returns:
            dict: Key is the name of the feature, value is the WoE Map.

        Raises:
            ValueError: If bin_target column has more than 2 categories.
        """
        if X[bin_target].nunique() != 2:
            raise ValueError(
                f"The target column '{bin_target}' must have exactly 2 unique values.")

        aux = X[[feature, bin_target]].copy().assign(n_row=1)
        aux = aux.pivot_table(index=feature, columns=bin_target,
                              values='n_row', aggfunc='sum', fill_value=0)
        aux /= aux.sum()
        aux['woe'] = np.log(aux[0] / aux[1])
        aux = aux.drop(columns=[0, 1])
        return {feature: aux['woe'].to_dict()}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Performs WoE transformation.

        Args:
            X (pd.DataFrame): Discrete data to be transformed.

        Raises:
            Exception: If fit method not called previously.

        Returns:
            pd.DataFrame: WoE encoded data.
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')

        aux = X.copy()
        for feature, woe_map in self._woe_encoding_map.items():
            aux[feature] = aux[feature].replace(woe_map)
        return aux

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Performs Inverse WoE transformation.

        Args:
            X (pd.DataFrame): WoE data to be transformed.

        Raises:
            Exception: If fit method not called previously.

        Returns:
            pd.DataFrame: WoE encoded data.
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')

        aux = X.copy()
        self._woe_reverse_map = {feature: {v: k for k, v in woe_map.items(
        )} for feature, woe_map in self._woe_encoding_map.items()}
        for feature, woe_map in self._woe_reverse_map.items():
            aux[feature] = aux[feature].replace(woe_map)
        return aux
