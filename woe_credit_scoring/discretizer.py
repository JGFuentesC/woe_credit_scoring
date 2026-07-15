from typing import Dict, List
from multiprocessing import Pool
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture


class Discretizer:
    """
    Discretizer class for transforming continuous data into discrete bins.

    This class provides methods to fit a discretization model to continuous data
    and transform the data into discrete bins. Supports multiple discretization
    strategies including 'uniform', 'quantile', 'kmeans', and 'gaussian'.
    Uses parallel processing to speed up computation on large datasets.

    Attributes:
        min_segments (int): Minimum number of bins to create.
        max_segments (int): Maximum number of bins to create.
        strategy (str): Discretization strategy to use.
        X (pd.DataFrame): The input data used for fitting the model.
        features (List[str]): List of feature names in the input data.
        edges_map (List[Dict]): List of edge configurations per feature/bin.
        __is_fitted (bool): Flag indicating whether the model has been fitted.
    """

    def __init__(self, min_segments: int = 2, max_segments: int = 5, strategy: str = 'quantile') -> None:
        self.__is_fitted = False
        self.X = None
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.strategy = strategy
        self.features = None
        self.edges_map = {}

    def __repr__(self) -> str:
        return (f"Discretizer(min_segments={self.min_segments}, "
                f"max_segments={self.max_segments}, strategy='{self.strategy}')")

    def get_params(self, deep: bool = True) -> Dict:
        return {
            'min_segments': self.min_segments,
            'max_segments': self.max_segments,
            'strategy': self.strategy,
        }

    def set_params(self, **params) -> 'Discretizer':
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def _make_pool(func, params: List, threads: int) -> List:
        with Pool(threads) as pool:
            data = pool.starmap(func, params)
        return data

    def fit(self, X: pd.DataFrame, n_threads: int = 1) -> None:
        self.X = X.copy()
        self.features = list(self.X.columns)
        self.edges_map = self._make_pool(
            self._discretize,
            [(self.X, feat, nbins, self.strategy) for feat in self.features
             for nbins in range(self.min_segments, self.max_segments + 1)],
            threads=n_threads
        )
        self.__is_fitted = True

    @staticmethod
    def _discretize(X: pd.DataFrame, feature: str, nbins: int, strategy: str) -> Dict:
        aux = X[[feature]].copy()
        has_missing = aux[feature].isnull().any()
        if has_missing:
            nonmiss = aux.dropna().reset_index(drop=True)
        else:
            nonmiss = aux.copy()

        if strategy != 'gaussian':
            if nonmiss[feature].nunique() > 1:
                n_bins = min(nbins, nonmiss[feature].nunique())
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                kb.fit(nonmiss[[feature]])
                edges = list(kb.bin_edges_[0])
                return {'feature': feature, 'nbins': nbins, 'edges': [-np.inf] + edges[1:-1] + [np.inf]}
            else:
                edges = [-np.inf, np.inf]
                return {'feature': feature, 'nbins': nbins, 'edges': edges}
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
        aux = pd.cut(X[feature], bins=edges, include_lowest=True)
        aux = pd.Series(np.where(aux.isnull(), 'MISSING', aux)).to_frame().astype(str)
        discretized_feature_name = f'disc_{feature}_{nbins}_{strategy}'
        aux.columns = [discretized_feature_name]
        return aux

    def transform(self, X: pd.DataFrame, n_threads: int = 1) -> pd.DataFrame:
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')

        if not self.edges_map:
            return pd.DataFrame(index=X.index)

        features = list(set(edge['feature'] for edge in self.edges_map))
        non_present_features = [f for f in features if f not in X.columns]
        if non_present_features:
            raise Exception(f"Missing features: {', '.join(non_present_features)}")

        encoded_data = self._make_pool(
            self._encode,
            [(X, edge_map['feature'], edge_map['nbins'], edge_map['edges'], self.strategy)
             for edge_map in self.edges_map],
            threads=n_threads
        )

        result = pd.concat(encoded_data, axis=1)
        return result
