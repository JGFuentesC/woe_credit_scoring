import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from .WoeBaseFeatureSelector import WoeBaseFeatureSelector
from .Discretizer import Discretizer

class WoeContinuousFeatureSelector(WoeBaseFeatureSelector):
    """
    Class for selecting continuous features based on their WoE transformation and 
    Information Value statistic.
    """
    selected_features: Optional[List[Dict[str, Union[str, float]]]] = None
    __is_fitted: bool = False
    _Xd: Optional[pd.DataFrame] = None
    discretizers: Optional[List[Discretizer]] = None
    iv_report: Optional[pd.DataFrame] = None

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = 'quantile', iv_threshold: float = 0.1, 
            min_bins: int = 2, max_bins: int = 5, n_threads: int = 1, strictly_monotonic: bool = False,bin_target:str='binary_target') -> None:
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
        
        Reference:
            For more information about the dcc and dec methods please refer to the following paper:
            https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0289130
        """
        
        cont_features = list(X.columns)
        methods = ['quantile', 'uniform', 'kmeans', 'gaussian']
        
        if method in methods:
            disc = Discretizer(strategy=method, min_segments=min_bins, max_segments=max_bins)
            disc.fit(X[cont_features], n_threads=n_threads)
            self._Xd = disc.transform(X[cont_features])
            self._Xd[bin_target] = y
            self.discretizers = [disc]
        elif method in ('dcc', 'dec'):
            discretizers = [Discretizer(strategy=m, min_segments=min_bins, max_segments=max_bins) for m in methods]
            for disc in discretizers:
                disc.fit(X[cont_features], n_threads=n_threads)
            self.discretizers = discretizers
            self._Xd = pd.concat([disc.transform(X[cont_features]) for disc in discretizers], axis=1)
            self._Xd[bin_target] = y
        else:
            raise Exception('Invalid method, options are quantile, uniform, kmeans, gaussian, dcc and dec')
        
        disc_features = list(self._Xd.columns)
        mono = None
        
        if strictly_monotonic:
            mono = {feature: self._check_monotonic(self._Xd[feature], self._Xd[bin_target]) for feature in disc_features}
            mono = {x: y for x, y in mono.items() if y}
            if not mono:
                raise Exception('There is no monotonic feature.\n Please try turning strictly_monotonic parameter to False or increase the number of bins')
            disc_features = list(mono.keys())
        
        iv = [(feature, self._information_value(self._Xd[feature], self._Xd[bin_target])) for feature in disc_features]
        self.iv_report = pd.DataFrame(iv, columns=['feature', 'iv'])
        self.iv_report['relevant'] = self.iv_report['iv'] >= iv_threshold
        iv = [(feature, value) for feature, value in iv if value >= iv_threshold]
        iv = pd.DataFrame(iv, columns=['feature', 'iv'])
        iv['root_feature'] = iv['feature'].apply(lambda x: "_".join(x.split('_')[1:-2]))
        iv['nbins'] = iv['feature'].apply(lambda x: x.split('_')[-2])
        iv['method'] = iv['feature'].apply(lambda x: x.split('_')[-1])
        
        if method in methods + ['dcc']:
            iv = iv.sort_values(by=['root_feature', 'iv', 'nbins'], ascending=[True, False, True]).reset_index(drop=True)
            iv['index'] = iv.groupby('root_feature').cumcount() + 1
        elif method == 'dec':
            iv = iv.sort_values(by=['root_feature', 'method', 'iv', 'nbins'], ascending=[True, True, False, True]).reset_index(drop=True)
            iv['index'] = iv.groupby(['root_feature', 'method']).cumcount() + 1
        
        iv = iv.loc[iv['index'] == 1].reset_index(drop=True)
        self.iv_report['selected'] = self.iv_report['feature'].isin(iv['feature'])
        self.iv_report.sort_values(by=['selected', 'relevant'], ascending=[False, False], inplace=True)
        cont_features = list(set(iv['root_feature']))
        
        for disc in self.discretizers:
            disc.fit(X[cont_features], n_threads=n_threads)
        
        self.selected_features = iv.drop('index', axis=1).to_dict(orient='records')
        self.__is_fitted = True
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts continuous features to their best discretization.

        Args:
            X (pd.DataFrame): Continuous predictors data.

        Raises:
            Exception: If fit method is not called first.
            Exception: If a fitted feature is not present in data.

        Returns:
            pd.DataFrame: Best discretization transformed data.
        """
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        
        aux = X.copy()
        features = list(set([feature['root_feature'] for feature in self.selected_features]))
        non_present_features = [f for f in features if f not in X.columns]
        
        if non_present_features:
            print(f'{", ".join(non_present_features)} feature{"s" if len(non_present_features) > 1 else ""} not present in data')
            raise Exception("Missing features")
        
        aux = pd.concat([disc.transform(X[features]) for disc in self.discretizers], axis=1)
        aux = aux[[feature['feature'] for feature in self.selected_features]]
        return aux
