from typing import Dict
from collections import ChainMap
from itertools import repeat
import numpy as np
import pandas as pd

class DiscreteNormalizer:
    """
    DiscreteNormalizer is a class for normalizing discrete data based on a specified relative frequency threshold.

    This class provides methods to fit a normalization model to discrete data and transform the data according to the learned normalization mapping.
    It handles missing values by assigning them to a specific category and groups infrequent categories into a default category.
    If the default category does not meet the relative frequency threshold, it is mapped to the most frequent category.

    Attributes:
        MISSING_VALUE (str): Placeholder for missing values.
        DEFAULT_THRESHOLD (float): Default threshold for considering a category as relevant.
        normalization_threshold (float): Threshold for considering a category as relevant.
        default_category (str): Name for the default grouping/new categories.
        normalization_map (dict): Mapping of original categories to normalized categories.
        features (list): List of feature names in the input data.
        new_categories (dict): Dictionary of new categories identified during transformation.
        X (pd.DataFrame): The input data used for fitting the model.
        __is_fitted (bool): Flag indicating whether the model has been fitted.

    Methods:
        fit(X): Learns discrete normalization mapping from the input data.
        transform(X): Transforms discrete data into its normalized form.
        _prepare_feature(feature): Prepares a feature by filling missing values and converting to string.
        _get_normalization_map(X, feature, threshold, default_category): Creates the normalization map for a given feature.
    """
    MISSING_VALUE = 'MISSING'
    DEFAULT_THRESHOLD = 0.05

    def __init__(self, normalization_threshold: float = DEFAULT_THRESHOLD, default_category: str = 'OTHER') -> None:
        """
        Args:
            normalization_threshold (float, optional): Threshold for considering a category as relevant. Defaults to 0.05.
            default_category (str, optional): Given name for the default grouping/new categories. Defaults to 'OTHER'.
        """
        self.__is_fitted = False
        self.normalization_threshold = normalization_threshold
        self.default_category = default_category
        self.normalization_map = None
        self.features = None
        self.new_categories = {}
        self.X = None

    def fit(self, X: pd.DataFrame) -> None:
        """Learns discrete normalization mapping taking into account the following rules:
            1. All missing values will be filled with the category 'MISSING'
            2. Categories which relative frequency is less than normalization threshold will be mapped to default_category
            3. If default_category as a group doesn't reach the relative frequency threshold, then it will be mapped to the most frequent category

        Args:
            X (pd.DataFrame): Data to be normalized

        Raises:
            TypeError: If provided data is not a pandas DataFrame object
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Please use a Pandas DataFrame object')

        self.X = X.copy()
        self.features = list(self.X.columns)
        self.normalization_map = {}

        for feat in self.features:
            self.X[feat] = self._prepare_feature(self.X[feat])

        self.normalization_map = dict(ChainMap(*map(
            lambda feat: self._get_normalization_map(
                self.X, feat, self.normalization_threshold, self.default_category),
            self.features
        )))
        self.__is_fitted = True

    @staticmethod
    def _prepare_feature(feature: pd.Series) -> pd.Series:
        """Prepares a feature by filling missing values and converting to string."""
        return feature.fillna(DiscreteNormalizer.MISSING_VALUE).astype(str)

    @staticmethod
    def _get_normalization_map(X: pd.DataFrame, feature: str, threshold: float, default_category: str) -> Dict:
        """Creates the normalization map and the list of existing categories for a given feature.

        Args:
            X (pd.DataFrame): Data with discrete features
            feature (str): Feature to be analyzed
            threshold (float): Threshold for considering a category as relevant. Defaults to 0.05.
            default_category (str): Given name for the default grouping/new categories. Defaults to 'OTHER'.

        Returns:
            dict: Feature is the key and value is a dictionary which keys are the replacement map and the list of existing categories.
        """
        aux = X[feature].value_counts(normalize=True).to_frame()
        aux.columns = [feature]
        aux['mapping'] = np.where(
            aux[feature] < threshold, default_category, aux.index)
        mode = aux.head(1)['mapping'].values[0]

        if aux.loc[aux['mapping'] == default_category][feature].sum() < threshold:
            aux['mapping'] = aux['mapping'].replace({default_category: mode})

        aux = aux.drop(feature, axis=1)
        return {
            feature: {
                'replacement_map': aux.loc[aux.index != aux['mapping']]['mapping'].to_dict(),
                'existing_categories': list(aux.index),
                'mode': mode
            }
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms discrete data into its normalized form.

        Args:
            X (pd.DataFrame): Data to be transformed

        Raises:
            Exception: If fit method not called previously
            Exception: If features analyzed during fit are not present in X

        Returns:
            pd.DataFrame: Normalized discrete data
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')

        aux = X.copy()
        features = list(self.normalization_map.keys())
        non_present_features = [f for f in features if f not in X.columns]

        if non_present_features:
            raise Exception(
                f"Missing features: {', '.join(non_present_features)}")

        for feat in features:
            aux[feat] = self._prepare_feature(aux[feat])
            mapping = self.normalization_map[feat]['replacement_map']
            existing_categories = self.normalization_map[feat]['existing_categories']
            new_categories = [
                cat for cat in aux[feat].unique() if cat not in existing_categories]

            if new_categories:
                self.new_categories.update({feat: new_categories})
                replacement = self.default_category if self.default_category in existing_categories else self.normalization_map[
                    feat]['mode']
                aux[feat] = aux[feat].replace(
                    dict(zip(new_categories, repeat(replacement))))

            aux[feat] = aux[feat].replace(mapping)

        return aux
