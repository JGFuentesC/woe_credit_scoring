from typing import Dict, List, Union, Tuple, Optional
from multiprocessing import Pool
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
import logging
from .base import WoeBaseFeatureSelector
from .normalizer import DiscreteNormalizer

logger = logging.getLogger("CreditScoringToolkit")

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
        self.iv_report = self.iv_report.sort_values('selected', ascending=False)
        disc_features = list(self.iv_report.loc[self.iv_report['selected']]['feature'])
        if len(disc_features) == 0:
            raise Exception(
                'No relevant feature found. Please try increasing the IV threshold')
        self.selected_features: dict[str, float] =self.iv_report.loc[self.iv_report['selected']].set_index('feature')[
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


class IVCalculator:
    """
    A class to calculate the Information Value (IV) for both discrete and continuous features.
    It provides a simple interface that abstracts away the manual steps of discretization and normalization.

    Example:
        >>> from woe_credit_scoring import IVCalculator
        >>> import pandas as pd
        >>> data = pd.read_csv('example_data/hmeq.csv')
        >>> iv_calculator = IVCalculator(
        ...     data=data,
        ...     target='BAD',
        ...     continuous_features=['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'],
        ...     discrete_features=['REASON', 'JOB']
        ... )
        >>> iv_report = iv_calculator.calculate_iv()
        >>> print(iv_report)

    """
    def __init__(self, data: pd.DataFrame, target: str, continuous_features: List[str] = None, discrete_features: List[str] = None):
        """
        Initializes the IVCalculator object.

        Args:
            data (pd.DataFrame): The input data containing features and target.
            target (str): The target variable name.
            continuous_features (List[str], optional): List of continuous feature names. Defaults to None.
            discrete_features (List[str], optional): List of discrete feature names. Defaults to None.
        """
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

    def calculate_iv(self,
                     max_discretization_bins: int = 5,
                     strictly_monotonic: bool = False,
                     discretization_method: str = 'quantile',
                     n_threads: int = 1,
                     discrete_normalization_threshold: float = 0.05,
                     discrete_normalization_default_category: str = 'OTHER'
                     ) -> pd.DataFrame:
        """
        Calculates the Information Value (IV) for the provided features.

        Args:
            max_discretization_bins (int, optional): The maximum number of bins for discretization. Defaults to 5.
            strictly_monotonic (bool, optional): Whether to enforce strictly monotonic WoE transformation for continuous features. Defaults to False.
            discretization_method (str, optional): The method for discretization ('quantile', 'uniform', 'kmeans', 'gaussian', 'dcc', 'dec'). Defaults to 'quantile'.
            n_threads (int, optional): The number of threads to use for parallel processing. Defaults to 1.
            discrete_normalization_threshold (float, optional): The threshold for discrete feature normalization. Defaults to 0.05.
            discrete_normalization_default_category (str, optional): The default category for discrete feature normalization. Defaults to 'OTHER'.

        Returns:
            pd.DataFrame: A DataFrame containing the IV report for all features, sorted by IV in descending order.
        """
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
                    iv_threshold=-np.inf,  # Using a very low threshold to get IV for all features
                    method=discretization_method,
                    n_threads=n_threads
                )
                iv_report_continuous = woe_continuous_selector.iv_report
                iv_report_continuous = iv_report_continuous[['root_feature', 'iv']].rename(columns={'root_feature': 'feature'})
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
                    iv_threshold=-np.inf  # Using a very low threshold to get IV for all features
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

        final_iv_report = pd.concat(iv_reports, axis=0).sort_values('iv', ascending=False).reset_index(drop=True)
        return final_iv_report
