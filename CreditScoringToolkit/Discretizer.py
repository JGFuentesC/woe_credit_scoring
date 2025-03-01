import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
from functools import reduce
from multiprocessing import Pool
from typing import List, Tuple, Dict, Union

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
            [(self.X, feat, nbins, self.strategy) for feat in self.features for nbins in range(self.min_segments, self.max_segments + 1)],
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
            nonmiss = aux.dropna()
        else:
            nonmiss = aux.copy()
        
        if strategy != 'gaussian':
            kb = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy=strategy)
            kb.fit(nonmiss[[feature]])
            edges = list(kb.bin_edges_[0])
            return {'feature': feature, 'nbins': nbins, 'edges': [-np.inf] + edges[1:-1] + [np.inf]}
        else:
            gm = GaussianMixture(n_components=nbins)
            gm.fit(nonmiss[[feature]])
            nonmiss['cluster'] = gm.predict(nonmiss[[feature]])
            edges = nonmiss.groupby('cluster')[feature].agg(['min', 'max']).sort_values(by='min')
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
        aux = pd.Series(np.where(aux.isnull(), 'MISSING', aux)).to_frame().astype(str)
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
            raise Exception('Please call fit method first with the required parameters')
        
        aux = X.copy()
        features = list(set(edge['feature'] for edge in self.edges_map))
        non_present_features = [f for f in features if f not in X.columns]
        if non_present_features:
            raise Exception(f"Missing features: {', '.join(non_present_features)}")
        
        encoded_data = self._make_pool(
            self._encode,
            [(X, edge_map['feature'], edge_map['nbins'], edge_map['edges'], self.strategy) for edge_map in self.edges_map],
            threads=n_threads
        )
        return reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='inner'), encoded_data).copy()
