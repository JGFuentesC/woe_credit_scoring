import numpy as np
import pandas as pd
from collections import ChainMap
from functools import reduce
from itertools import repeat
from multiprocessing import Pool
from typing import Dict, List, Union, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer

from varclushi import VarClusHi
from scipy.stats.mstats import winsorize

import logging

logger = logging.getLogger("CreditScoringToolkit")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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

        aux.drop(feature, axis=1, inplace=True)
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


def frequency_table(df: pd.DataFrame, variables: Union[List[str], str]) -> None:
    """
    Displays a frequency table for the specified variables in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variables (Union[List[str], str]): List of variables (column names) to generate frequency tables for.

    Returns:
        None
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The first argument must be a pandas DataFrame.")

    if isinstance(variables, str):
        variables = [variables]

    if not isinstance(variables, list) or not all(isinstance(var, str) for var in variables):
        raise TypeError(
            "The second argument must be a string or a list of strings.")

    for variable in variables:
        if variable not in df.columns:
            logger.warning(f"{variable} not found in DataFrame columns.")
            continue

        frequency_df = df[variable].value_counts().to_frame().sort_index()
        frequency_df.columns = ['Abs. Freq.']
        frequency_df['Rel. Freq.'] = frequency_df['Abs. Freq.'] / \
            frequency_df['Abs. Freq.'].sum()
        frequency_df[['Cum. Abs. Freq.', 'Cum. Rel. Freq.']
                     ] = frequency_df.cumsum()

        print(f'**** Frequency Table for {variable} ****\n')
        print(frequency_df)
        print("\n" * 3)


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
        aux['woe'] = np.log(aux[0] / aux[1])
        aux['iv'] = (aux[0] - aux[1]) * aux['woe']
        iv = aux['iv'].sum()
        return None if np.isinf(iv) else iv

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


class Discretizer:
    """
    Discretizer class for transforming continuous data into discrete bins.

    This class provides methods to fit a discretization model to continuous data and transform the data into discrete bins.
    It supports multiple discretization strategies including 'uniform', 'quantile', 'kmeans', and 'gaussian'.
    The class uses parallel processing to speed up the computation when dealing with large datasets.

    Attributes:
        min_segments (int): Minimum number of bins to create.
        max_segments (int): Maximum number of bins to create.
        strategy (str): Discretization strategy to use.
        X (pd.DataFrame): The input data used for fitting the model.
        features (List[str]): List of feature names in the input data.
        edges_map (Dict): Dictionary mapping features to their respective bin edges.
        __is_fitted (bool): Flag indicating whether the model has been fitted.

    Methods:
        _make_pool(func, params, threads): Executes a function with a set of parameters using pooling threads.
        fit(X, n_threads): Learns discretization edges from the input data.
        transform(X, n_threads): Transforms continuous data into its discrete form.
        _discretize(X, feature, nbins, strategy): Discretizes a series into a specified number of bins using the given strategy.
        _encode(X, feature, nbins, edges, strategy): Encodes a continuous feature into a discrete bin.
    """

    def __init__(self, min_segments: int = 2, max_segments: int = 5, strategy: str = 'quantile') -> None:
        self.__is_fitted = False
        self.X = None
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.strategy = strategy
        self.features = None
        self.edges_map = {}

    @staticmethod
    def _make_pool(func, params: List[Tuple], threads: int) -> List:
        """
        Executes a function with a set of parameters using pooling threads.

        Args:
            func (function): Function to be executed.
            params (list): List of tuples, each tuple is a parameter combination.
            threads (int): Number of pooling threads to use.

        Returns:
            list: All execution results in a list.
        """
        with Pool(threads) as pool:
            data = pool.starmap(func, params)
        return data

    def fit(self, X: pd.DataFrame, n_threads: int = 1) -> None:
        """
        Learns discretization edges.

        Args:
            X (pd.DataFrame): Data to be discretized.
            n_threads (int, optional): Number of pooling threads. Defaults to 1.
        """
        self.X = X.copy()
        self.features = list(self.X.columns)
        self.edges_map = self._make_pool(
            self._discretize,
            [(self.X, feat, nbins, self.strategy) for feat in self.features for nbins in range(
                self.min_segments, self.max_segments + 1)],
            threads=n_threads
        )
        self.__is_fitted = True

    @staticmethod
    def _discretize(X: pd.DataFrame, feature: str, nbins: int, strategy: str) -> Dict:
        """
        Discretizes a series in a particular number of bins using the given strategy.

        Args:
            X (pd.DataFrame): Data to be discretized.
            feature (str): Feature name.
            nbins (int): Number of expected bins.
            strategy (str): {'uniform', 'quantile', 'kmeans', 'gaussian'}, discretization method to be used.

        Returns:
            dict: Discretized data.

        Reference:
            For more details on the discretization strategies, see
            https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0289130
        """
        aux = X[[feature]].copy()
        has_missing = aux[feature].isnull().any()
        if has_missing:
            nonmiss = aux.dropna().reset_index(drop=True)
        else:
            nonmiss = aux.copy()

        if strategy != 'gaussian':
            kb = KBinsDiscretizer(
                n_bins=nbins, encode='ordinal', strategy=strategy)
            kb.fit(nonmiss[[feature]])
            edges = list(kb.bin_edges_[0])
            return {'feature': feature, 'nbins': nbins, 'edges': [-np.inf] + edges[1:-1] + [np.inf]}
        else:
            gm = GaussianMixture(n_components=nbins)
            gm.fit(nonmiss[[feature]])
            nonmiss['cluster'] = gm.predict(nonmiss[[feature]])
            edges = nonmiss.groupby('cluster')[feature].agg(
                ['min', 'max']).sort_values(by='min')
            edges = sorted(set(edges['min'].tolist() + edges['max'].tolist()))
            return {'feature': feature, 'nbins': nbins, 'edges': [-np.inf] + edges[1:-1] + [np.inf]}

    @staticmethod
    def _encode(X: pd.DataFrame, feature: str, nbins: int, edges: List[float], strategy: str) -> pd.DataFrame:
        """
        Encodes continuous feature into a discrete bin.

        Args:
            X (pd.DataFrame): Continuous data.
            feature (str): Feature to be encoded.
            nbins (int): Number of encoding bins.
            edges (list): Bin edges list.
            strategy (str): {'uniform', 'quantile', 'kmeans', 'gaussian'}, discretization strategy.

        Returns:
            pd.DataFrame: Encoded data.
        """
        aux = pd.cut(X[feature], bins=edges, include_lowest=True)
        aux = pd.Series(np.where(aux.isnull(), 'MISSING', aux)
                        ).to_frame().astype(str)
        discretized_feature_name = f'disc_{feature}_{nbins}_{strategy}'
        aux.columns = [discretized_feature_name]
        return aux

    def transform(self, X: pd.DataFrame, n_threads: int = 1) -> pd.DataFrame:
        """
        Transforms continuous data into its discrete form.

        Args:
            X (pd.DataFrame): Data to be discretized.
            n_threads (int, optional): Number of pooling threads to speed computation. Defaults to 1.

        Raises:
            Exception: If fit method not called previously.
            Exception: If features analyzed during fit are not present in X.

        Returns:
            pd.DataFrame: Discretized Data.
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')

        aux = X.copy()
        features = list(set(edge['feature'] for edge in self.edges_map))
        non_present_features = [f for f in features if f not in X.columns]
        if non_present_features:
            raise Exception(
                f"Missing features: {', '.join(non_present_features)}")

        encoded_data = self._make_pool(
            self._encode,
            [(X, edge_map['feature'], edge_map['nbins'], edge_map['edges'],
              self.strategy) for edge_map in self.edges_map],
            threads=n_threads
        )

        result = reduce(lambda x, y: pd.merge(
            x, y, left_index=True, right_index=True, how='inner'), encoded_data).copy()
        return result


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
        aux.drop(columns=[0, 1], inplace=True)
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


class WoeContinuousFeatureSelector(WoeBaseFeatureSelector):
    """
    WoeContinuousFeatureSelector is a class for selecting continuous features based on their Weight of Evidence (WoE) transformation and Information Value (IV) statistic.

    This class provides methods to fit a model that evaluates continuous features by discretizing them into bins, transforming them using WoE, and calculating their IV. 
    It supports multiple discretization strategies including 'quantile', 'uniform', 'kmeans', 'gaussian', 'dcc', and 'dec'. 
    The class can also enforce monotonic risk behavior for the selected features if required.

    Attributes:
        selected_features (Optional[List[Dict[str, Union[str, float]]]]): List of selected features with their respective IV values.
        __is_fitted (bool): Flag indicating whether the model has been fitted.
        _Xd (Optional[pd.DataFrame]): DataFrame containing the discretized features.
        discretizers (Optional[List[Discretizer]]): List of Discretizer objects used for discretizing the features.
        iv_report (Optional[pd.DataFrame]): DataFrame containing the IV report for all features.

    Methods:
        fit(X, y, method, iv_threshold, min_bins, max_bins, n_threads, strictly_monotonic): Learns the best features given an IV threshold and optional monotonic risk restriction.
        transform(X): Converts continuous features to their best discretization.
    """
    selected_features: Optional[List[Dict[str, Union[str, float]]]] = None
    __is_fitted: bool = False
    _Xd: Optional[pd.DataFrame] = None
    discretizers: Optional[List[Discretizer]] = None
    iv_report: Optional[pd.DataFrame] = None

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = 'quantile', iv_threshold: float = 0.1,
            min_bins: int = 2, max_bins: int = 5, n_threads: int = 1, strictly_monotonic: bool = False) -> None:
        """
        Learns the best features given an IV threshold. Monotonic risk restriction can be applied.

        Args:
            X (pd.DataFrame): Predictors data.
            y (pd.Series): Dichotomic response feature.
            method (str, optional): Discretization technique. Options are {'quantile', 'uniform', 'kmeans', 'gaussian', 'dcc', 'dec'}.
                Defaults to 'quantile'.
            iv_threshold (float, optional): IV value for a feature to be included in final selection. Defaults to 0.1.
            min_bins (int, optional): Minimum number of discretization bins. Defaults to 2.
            max_bins (int, optional): Maximum number of discretization bins. Defaults to 5.
            n_threads (int, optional): Number of multiprocessing threads. Defaults to 1.
            strictly_monotonic (bool, optional): Indicates if only monotonic risk features should be selected. Defaults to False.

        Raises:
            Exception: If strictly_monotonic=True and no monotonic feature is present in the final selection.
            Exception: If method is not in {'quantile', 'uniform', 'kmeans', 'gaussian', 'dcc', 'dec'}.
            Exception: If X is not a pandas DataFrame.
            Exception: If y is not a pandas Series.

        Reference:
            For more information about the dcc and dec methods please refer to the following paper:
            https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0289130
        """
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
            mono = {feature: self._check_monotonic(self._Xd[feature], self._Xd['binary_target']) for feature in disc_features}
            mono = {x: y for x, y in mono.items() if y}
            if not mono:
                raise Exception('There is no monotonic feature.\n Please try turning strictly_monotonic parameter to False or increase the number of bins')
            disc_features = list(mono.keys())

        iv = [(feature, self._information_value(self._Xd[feature], self._Xd['binary_target'])) for feature in disc_features]
        self.iv_report = pd.DataFrame(iv, columns=['feature', 'iv']).dropna().reset_index(drop=True)
        self.iv_report['relevant'] = self.iv_report['iv'] >= iv_threshold
        
        self.iv_report['root_feature'] = self.iv_report['feature'].apply(lambda x: "_".join(x.split('_')[1:-2]))
        self.iv_report['nbins'] = self.iv_report['feature'].apply(lambda x: x.split('_')[-2])
        self.iv_report['method'] = self.iv_report['feature'].apply(lambda x: x.split('_')[-1])

        sort_columns = ['root_feature', 'iv', 'nbins'] if method in methods + ['dcc'] else ['root_feature', 'method', 'iv', 'nbins']
        self.iv_report = self.iv_report.sort_values(by=sort_columns, ascending=[True, False, True] if method in methods + ['dcc'] else [True, True, False, True]).reset_index(drop=True)
        self.iv_report['index'] = self.iv_report.groupby('root_feature').cumcount() + 1 if method in methods + ['dcc'] else self.iv_report.groupby(['root_feature', 'method']).cumcount() + 1

        self.iv_report = self.iv_report.loc[self.iv_report['index'] == 1].reset_index(drop=True)
        self.iv_report['selected'] = self.iv_report['feature'].isin(self.iv_report['feature'])
        self.iv_report = self.iv_report.sort_values(by=['selected', 'relevant'], ascending=[False, False])
        cont_features = list(set(self.iv_report.loc[self.iv_report['relevant']]['root_feature']))
        if len(cont_features) == 0:
            raise Exception('No relevant feature found. Please try increasing the number of bins or changing the discretization method')
        for disc in self.discretizers:
            disc.fit(X[cont_features], n_threads=n_threads)
        self.selected_features =self.iv_report[self.iv_report['relevant']].drop('index', axis=1).to_dict(orient='records')
        self.__is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts continuous features to their best discretization.

        Args:
            X (pd.DataFrame): Continuous predictors data.

        Raises:
            Exception: If fit method is not called first.
            Exception: If a fitted feature is not present in data.
            Exception: If X is not a pandas DataFrame.

        Returns:
            pd.DataFrame: Best discretization transformed data.
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')

        aux = X.copy()
        features = list(set([feature['root_feature']
                        for feature in self.selected_features]))
        non_present_features = [f for f in features if f not in X.columns]

        if non_present_features:
            logger.exception(f'{", ".join(non_present_features)} feature{"s" if len(non_present_features) > 1 else ""} not present in data')
            raise Exception("Missing features")

        aux = pd.concat([disc.transform(X[features])
                        for disc in self.discretizers], axis=1)
        aux = aux[[feature['feature'] for feature in self.selected_features]]
        return aux


class WoeDiscreteFeatureSelector(WoeBaseFeatureSelector):
    """
    WoeDiscreteFeatureSelector is a class for selecting discrete features based on their Weight of Evidence (WoE) 
    transformation and Information Value (IV) statistic. This class inherits from WoeBaseFeatureSelector and provides 
    methods to fit the model to the data and transform the data by keeping only the selected features.

    The fit method evaluates each feature's predictive power by calculating its IV and selects features that meet 
    a specified IV threshold. The transform method then filters the dataset to include only these selected features.

    Attributes:
        iv_report (pd.DataFrame): A DataFrame containing the IV values and selection status of each feature.
        selected_features (dict[str, float]): A dictionary of selected features and their corresponding IV values.
        __is_fitted (bool): A flag indicating whether the fit method has been called.
    """
    iv_report: pd.DataFrame = None

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series, iv_threshold: float = 0.1) -> None:
        """Learns best features given an IV threshold.

        Args:
            X (pd.DataFrame): Discrete predictors data
            y (pd.Series): Dichotomic response feature
            iv_threshold (float, optional):  IV value for a feature to be included in final selection. Defaults to 0.1.
        """
        disc_features: list[str] = list(X.columns)
        aux: pd.DataFrame = X.copy()
        aux['binary_target'] = y
        iv: list[tuple[str, float]] = [(feature, self._information_value(
            aux[feature], aux['binary_target'])) for feature in disc_features]
        self.iv_report = pd.DataFrame(iv, columns=['feature', 'iv']).dropna().reset_index(drop=True)
        self.iv_report['selected'] = self.iv_report['iv'] >= iv_threshold
        self.iv_report.sort_values('selected', ascending=False, inplace=True)
        iv = [(feature, value)
              for feature, value in iv if value >= iv_threshold]
        iv = pd.DataFrame(iv, columns=['feature', 'iv'])
        disc_features = list(iv['feature'])
        if len(disc_features) == 0:
            raise Exception(
                'No relevant feature found. Please try increasing the IV threshold')
        self.selected_features: dict[str, float] = iv.set_index('feature')[
            'iv'].to_dict()
        self.__is_fitted: bool = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms data keeping only the selected features

        Args:
            X (pd.DataFrame): Discrete predictors data

        Raises:
            Exception: If fit method is not called first.
            Exception: If a fitted feature is not present in data.

        Returns:
            pd.DataFrame: Data containing best discrete features 
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')
        else:
            aux: pd.DataFrame = X.copy()
            features: list[str] = [
                feature for feature in self.selected_features.keys()]
            non_present_features: list[str] = [
                f for f in features if f not in X.columns]
            if len(non_present_features) > 0:
                logger.exception(
                    f'{",".join(non_present_features)} feature{"s" if len(non_present_features) > 1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                aux = aux[features]
                return aux


class CreditScoring:
    """
    Implements credit risk scorecards following the methodology proposed in 
    Siddiqi, N. (2012). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.

    This class provides methods to fit a logistic regression model to the provided data,
    transform the data using Weight of Evidence (WoE) encoding, and generate a scorecard
    that maps the model's coefficients to a scoring system. The scorecard can then be used
    to convert new data into credit scores.

    Attributes:
        logistic_regression (Optional[LogisticRegression]): Fitted logistic regression model.
        pdo (Optional[int]): Points to Double the Odds.
        base_odds (Optional[int]): Base odds at the base score.
        base_score (Optional[int]): Base score for calibration.
        betas (Optional[list]): Coefficients of the logistic regression model.
        alpha (Optional[float]): Intercept of the logistic regression model.
        factor (Optional[float]): Factor used in score calculation.
        offset (Optional[float]): Offset used in score calculation.
        features (Optional[Dict[str, float]]): Mapping of feature names to their coefficients.
        n (Optional[int]): Number of features.
        scorecard (Optional[pd.DataFrame]): DataFrame containing the scorecard.
        scoring_map (Optional[Dict[str, Dict[str, int]]]): Mapping of features to their score mappings.
        __is_fitted (bool): Indicates whether the model has been fitted.
    """

    logistic_regression: Optional[LogisticRegression] = None
    pdo: Optional[int] = None
    base_odds: Optional[int] = None
    base_score: Optional[int] = None
    betas: Optional[list] = None
    alpha: Optional[float] = None
    factor: Optional[float] = None
    offset: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    n: Optional[int] = None
    scorecard: Optional[pd.DataFrame] = None
    scoring_map: Optional[Dict[str, Dict[str, int]]] = None
    __is_fitted: bool = False

    def __init__(self, pdo: int = 20, base_score: int = 400, base_odds: int = 1) -> None:
        """Initializes Credit Scoring object.

        Args:
            pdo (int, optional): Points to Double the Odd's _. Defaults to 20.
            base_score (int, optional): Default score for calibration. Defaults to 400.
            base_odds (int, optional): Odd's base at base_score . Defaults to 1.
        """
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)

    @staticmethod
    def _get_scorecard(X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Generates scorecard points for a given feature

        Args:
            X (pd.DataFrame): Feature Data
            feature (str): Predictor

        Returns:
            pd.DataFrame: Feature, Attribute and respective points
        """
        sc = X[[feature, f'P_{feature}']].copy(
        ).drop_duplicates().reset_index(drop=True)
        sc.rename(columns={feature: 'attribute',
                  f'P_{feature}': 'points'}, inplace=True)
        sc.insert(0, 'feature', feature)
        return sc

    def fit(self, Xw: pd.DataFrame, woe_encoder: WoeEncoder, logistic_regression: LogisticRegression) -> None:
        """Learns scoring map

        Args:
            Xw (pd.DataFrame): WoE transformed data
            woe_encoder (WoeEncoder): WoE encoder fitted object
            logistic_regression (LogisticRegression): Fitted logistic regression model
        """
        X = Xw.copy()
        self.betas = list(logistic_regression.coef_[0])
        self.alpha = logistic_regression.intercept_[0]
        self.features = dict(zip(Xw.columns, self.betas))
        self.n = len(self.betas)
        for feature, beta in self.features.items():
            X[f'P_{feature}'] = np.floor(
                (-X[feature] * beta + self.alpha / self.n) * self.factor + self.offset / self.n).astype(int)
        features = list(self.features.keys())
        X[features] = woe_encoder.inverse_transform(X[features])
        self.scorecard = pd.concat(
            map(lambda f: self._get_scorecard(X, f), features))
        self.scorecard = self.scorecard.groupby(['feature', 'attribute']).max()
        self.scoring_map = dict(ChainMap(*[{f: d[['attribute', 'points']].set_index('attribute')[
                                'points'].to_dict()} for f, d in self.scorecard.reset_index().groupby('feature')]))
        self.__is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Converts discrete data to scores

        Args:
            X (pd.DataFrame): Discrete predictor data

        Raises:
            Exception: If fit method is not called first.
            Exception: If a fitted feature is not present in data.

        Returns:
            pd.DataFrame: Total score and scores for each feature 
        """
        if not self.__is_fitted:
            raise Exception(
                'Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = list(self.scoring_map.keys())
            non_present_features = [
                f for f in features if f not in aux.columns]
            if len(non_present_features) > 0:
                logger.exception(
                    f'{",".join(non_present_features)} feature{"s" if len(non_present_features) > 1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                for feature, points_map in self.scoring_map.items():
                    aux[feature].replace(points_map, inplace=True)
                aux['score'] = aux[features].sum(axis=1)
                return aux
