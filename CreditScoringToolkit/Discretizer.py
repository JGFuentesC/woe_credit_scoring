import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
from functools import reduce
from multiprocessing import Pool

class Discretizer:
    """ 
        Class for discretizing continuous data into bins
    """
    __is_fitted = False 
    X = None
    min_segments = 2
    max_segments = 5
    strategy = 'quantile' 
    features = None
    edges_map = {} 

    def __init__(self,min_segments:int=2,max_segments:int=5,strategy:str='quantile') -> None:
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.strategy = strategy
    
    @staticmethod
    def make_pool(func, params:list, threads:int)->list:
        """executes a function with a set of parameters using pooling threads

        Args:
            func (function): function to be executed
            params (list): list of tuples, each tuple is a parameter combination 
            threads (int): number of pooling threads to use

        Returns:
            list: all execution results in a list 
        """
        pool = Pool(threads)
        data = pool.starmap(func, params)
        pool.close()
        pool.join()
        del pool
        return [x for x in data]

    def fit(self,X:pd.DataFrame,n_threads:int=1)->None:
        """Learns discretization edges 
        Args:
            X (pd.DataFrame): data to be discretized
            n_threads (int, optional): number of pooling threads. Defaults to 1.
        """
        self.X = X.copy()
        features = list(self.X.columns)
        self.edges_map = self.make_pool(self._discretize,[(self.X,feat,nbins,self.strategy,) for feat in features for nbins in range(self.min_segments,self.max_segments+1)],threads=n_threads)
        self.__is_fitted = True

    @staticmethod
    def _discretize(X:pd.DataFrame,feature:str,nbins:int,strategy:str)->pd.DataFrame:
        """Discretizes a series in a particular number of bins using the given strategy

        Args:
            X (pd.DataFrame): Data to be discretized
            feature (str): Feature name
            nbins (int): Number of expected bins 
            strategy (str): {'uniform','quantile','kmeans','gaussian'}, discretization method to be used.

        Returns:
            pd.DataFrame: Discretized data 
        """
        aux = X[[feature]].copy()
        
        _has_missing = len(aux[feature].isnull().value_counts())==2
        if _has_missing:
            nonmiss,_ = [data for _,data in aux.groupby(aux[feature].isnull())]
        else:
            nonmiss  = aux.copy()
        
        if strategy!='gaussian':
            kb = KBinsDiscretizer(n_bins=nbins,encode='ordinal',strategy=strategy)
            kb.fit(nonmiss[[feature]])
            edges = list(kb.bin_edges_[0])
            return {'feature':feature,'nbins':nbins,'edges':[-np.inf]+edges[1:-1]+[np.inf]}
        elif strategy == 'gaussian':
            kb = GaussianMixture(n_components=nbins)
            kb.fit(nonmiss[[feature]])
            nonmiss['cluster'] = kb.predict(nonmiss[[feature]])
            edges = nonmiss.groupby('cluster').agg(['min','max'])
            edges.columns = ['lower_bound','upper_bound']
            edges.sort_values(by='lower_bound',inplace=True)
            edges = list(edges['lower_bound'])+list(edges['upper_bound'])[-1:]
            edges = sorted(set(edges))
            return {'feature':feature,'nbins':nbins,'edges':[-np.inf]+edges[1:-1]+[np.inf]}

    @staticmethod
    def _encode(X:pd.DataFrame,feature:str,nbins:int,edges:list,strategy:str)->pd.DataFrame:
        """
        Encodes continuous feature into a discrete bin

        Args:
            X (pd.DataFrame): Continuous data
            feature (str): Feature to be encoded
            nbins (int): Number of encoding bins 
            edges (list): Bin edges list
            strategy (str): {'uniform','quantile','kmeans','gaussian'}, Discretization strategy 

        Returns:
            pd.DataFrame: Encoded data
        """
        aux = pd.cut(X[feature],bins=edges,include_lowest=True)
        aux = pd.Series(np.where(aux.isnull(),'MISSING',aux)).to_frame().astype(str)
        discretized_feature_name = f'disc_{feature}_{nbins}_{strategy}'
        aux.columns = [discretized_feature_name]
        return aux

    def transform(self,X:pd.DataFrame,n_threads:int=1)->pd.DataFrame:
        """Transforms continuous data into its discrete form

        Args:
            X (pd.DataFrame): Data to be discretized
            n_threads (int, optional): Number of pooling threads to speed computation. Defaults to 1.

        Raises:
            Exception: If fit method not called previously
            Exception: If features analyzed during fit are not present in X

        Returns:
            pd.DataFrame: Discretized Data
        """
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = list(set([edge['feature'] for edge in self.edges_map]))
            non_present_features = [f for f in features if f not in X.columns]
            if len(non_present_features)>0:
                print(f'{",".join(non_present_features)} feature{"s" if len(non_present_features)>1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                encoded_data = self.make_pool(self._encode,[(X,edge_map['feature'],edge_map['nbins'],edge_map['edges'],self.strategy,) for edge_map in self.edges_map] ,threads=n_threads)
                return reduce(lambda x,y:pd.merge(x,y,left_index=True,right_index=True,how='inner'),encoded_data).copy()            
