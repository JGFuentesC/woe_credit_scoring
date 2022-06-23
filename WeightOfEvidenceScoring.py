import numpy as np 
import pandas as pd
from collections import ChainMap
from functools import reduce
from itertools import repeat


from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

from multiprocessing import Pool

def frequency_table(df:pd.DataFrame,var:list):
    """Displays a frequency table 

    Args:
        df (pd.DataFrame): Data
        var (list): List of variables 
    """
    if type(var)==str:
        var = [var]
    for v in var:
        aux = df[v].value_counts().to_frame().sort_index()
        aux.columns = ['Abs. Freq.']
        aux['Rel. Freq.'] = aux['Abs. Freq.']/aux['Abs. Freq.'].sum()
        aux[['Cumm. Abs. Freq.','Cumm. Rel. Freq.']] = aux.cumsum()
        print(f'****Frequency Table  {v}  ***\n\n')
        print(aux)
        print("\n"*3)

class DiscreteNormalizer:
    """ 
        Class for normalizing discrete data for a given relative frequency threshold
    """
    __is_fitted = False 
    normalization_threshold = None
    normalization_map = None
    default_category = None
    features = None 
    new_categories = {}
    mode = None 
    X = None

    def __init__(self,normalization_threshold:float=0.05,default_category:str='OTHER') -> None:
        """
        Args:
            normalization_threshold (float, optional): Threshold for considering a category as relevant. Defaults to 0.05.
            default_category (str, optional): Given name for the default grouping/new categories . Defaults to 'OTHER'.
        """
        self.normalization_threshold = normalization_threshold
        self.default_category = default_category
        

    def fit(self,X:pd.DataFrame)->None:
        """Learns discrete normalization mapping taking into account the following rules:
            1. All missing values will be filled with the category 'MISSING'
            2. Categories which relative frequency is less than normalization threshold will be mapped to default_category
            3. If default_category as a group doesn't reach the relative frequency threshold, then it will be mapped to the most frequent category 

        Args:
            X (pd.DataFrame): Data to be normalized

        Raises:
            Exception: If provided data is not a pandas DataFrame object
        """
        if type(X)!=pd.DataFrame:
            raise Exception('Please use a Pandas DataFrame object') 
        else:
            self.X = X.copy()
            self.features = list(self.X.columns)
            self.normalization_map = {}
            for feat in self.features:
                self.X[feat]  = self.X[feat].fillna('MISSING').astype(str)
            self.normalization_map = dict(ChainMap(*map(lambda feat:self._get_normalization_map(self.X,feat,self.normalization_threshold,self.default_category),self.features)))
            self.__is_fitted = True
            

    @staticmethod
    def _get_normalization_map(X:pd.DataFrame,feature:str,threshold:float,default_category:str)->dict:
        """Creates the normalization map and the list of existing categories for a given feature 

        Args:
            X (pd.DataFrame): Data with discrete features
            feature (str): feature to be analyzed 
            threshold (float): Threshold for considering a category as relevant. Defaults to 0.05.
            default_category (str): Given name for the default grouping/new categories . Defaults to 'OTHER'.

        Returns:
            dict: feature is the key and value is a dictionary which keys are the replacement map and the list of existing
            categories. 
        """
       
        aux = X[feature].value_counts(1).to_frame()
        aux['mapping'] = np.where(aux[feature]<threshold,default_category,aux.index)
        mode = aux.head(1)['mapping'].values[0]
        if aux.loc[aux['mapping']==default_category][feature].sum()<threshold:
            aux['mapping'].replace({default_category:mode},inplace=True)
        aux.drop(feature,axis=1,inplace=True)
        return {feature:{'replacement_map':aux.loc[aux.index!=aux['mapping']]['mapping'].to_dict(),
        'existing_categories':list(aux.index),'mode':mode}}


    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        """Transforms Discrete data into its normalized form

        Args:
            X (pd.DataFrame): Data to be transformed

        Raises:
            Exception: If fit method not called previously
            Exception: If features analyzed during fit are not present in X

        Returns:
            pd.DataFrame: Normalized discrete data
        """
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = list(self.normalization_map.keys())
            non_present_features = [f for f in features if f not in X.columns]
            if len(non_present_features)>0:
                print(f'{",".join(non_present_features)} feature{"s" if len(non_present_features)>1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                for feat in features:
                    aux[feat]  = aux[feat].fillna('MISSING').astype(str)
                    mapping = self.normalization_map[feat]['replacement_map']
                    existing_categories = self.normalization_map[feat]['existing_categories'][:]
                    new_categories = [cat for cat in aux[feat].unique() if cat not in existing_categories]
                    if len(new_categories)>0:
                        self.new_categories.update({feat:new_categories})
                        if self.default_category in self.normalization_map[feat]['existing_categories']:
                            replacement = self.default_category
                        else:
                            replacement = self.normalization_map[feat]['mode']
                        aux[feat].replace(dict(zip(new_categories,repeat(replacement))),inplace=True)
                    aux[feat].replace(mapping,inplace=True)
                return aux 

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
            return {'feature':feature,'nbins':nbins,'edges':list(kb.bin_edges_[0])}
        elif strategy == 'gaussian':
            kb = GaussianMixture(n_components=nbins)
            kb.fit(nonmiss[[feature]])
            nonmiss['cluster'] = kb.predict(nonmiss[[feature]])
            edges = nonmiss.groupby('cluster').agg(['min','max'])
            edges.columns = ['lower_bound','upper_bound']
            edges.sort_values(by='lower_bound',inplace=True)
            edges = list(edges['lower_bound'])+list(edges['upper_bound'])[-1:]
            edges = sorted(set(edges))
            return {'feature':feature,'nbins':nbins,'edges':edges}

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

class WoeEncoder:
    """ 
        Class for encoding discrete features into Weight of Evidence(WoE) transformation
    """
    features = None
    _woe_encoding_map = None
    __is_fitted = False 
    _woe_reverse_map = None 
    def __init__(self) -> None:
        pass

    def fit(self,X:pd.DataFrame,y:pd.Series)->None:
        """Learns WoE encoding 

        Args:
            X (pd.DataFrame): Data with discrete feature
            y (pd.Series): Dichotomic response
        """
        aux = X.copy()
        self.features = list(aux.columns)

        aux['binary_target'] = y
        self._woe_encoding_map = dict(ChainMap(*map(lambda feature:self._woe_transformation(aux,feature,'binary_target'),self.features)))
        self.__is_fitted = True

    @staticmethod
    def _woe_transformation(X:pd.DataFrame,feature:str,bin_target:str)->dict:
        """Calculates WoE Map between discrete space and log odd's space

        Args:
            X (pd.DataFrame): Discrete data including dichotomic response feature
            feature (str): name of the feature for getting the map
            bin_target (str): name of the dichotomic response feature

        Returns:
            dict: key is name of the feature, value is the WoE Map
        """
        aux = X[[feature,bin_target]].copy().assign(n_row=1)
        aux = aux.pivot_table(index=feature,columns=bin_target,values='n_row',aggfunc='sum',fill_value=0)
        aux/=aux.sum()
        aux['woe'] = np.log(aux[0]/aux[1])
        aux.drop(range(2),axis=1,inplace=True)
        return {feature:aux['woe'].to_dict()}

    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        """Performs WoE transformation

        Args:
            X (pd.DataFrame): Discrete data to be transformed

        Raises:
            Exception: If fit method not called previously

        Returns:
            pd.DataFrame: WoE encoded data
        """
        aux = X.copy()
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        else:
            for feature,woe_map in self._woe_encoding_map.items():
                aux[feature] = aux[feature].replace(woe_map)
            return aux
    def inverse_transform(self,X:pd.DataFrame)->pd.DataFrame:
        """Performs Inverse WoE transformation

        Args:
            X (pd.DataFrame): WoE data to be transformed

        Raises:
            Exception: If fit method not called previously

        Returns:
            pd.DataFrame: WoE encoded data
        """
        if not self.__is_fitted:
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            self._woe_reverse_map = {x:{z:y for y,z in d.items()} for x,d in self._woe_encoding_map.items()}
            for feature,woe_map in self._woe_reverse_map.items():
                aux[feature] = aux[feature].replace(woe_map)
            return aux

class WoeBaseFeatureSelector:
    """
        Base class for selecting features based on their WoE transformation and 
        Information Value statistic. 
    """
    def __init__(self):
        pass

    @staticmethod
    def _information_value(X:pd.Series,y:pd.Series)->float:
        """Computes information value (IV) statistic

        Args:
            X (pd.Series): Discretized predictors data
            y (pd.Series): Dichotomic response feature

        Returns:
            float: IV statistic
        """
        aux = pd.concat([X,y],axis=1)
        aux.columns = ['x','y']
        aux = aux.assign(nrow=1)
        aux = aux.pivot_table(index='x',columns='y',values='nrow',aggfunc='sum',fill_value=0)
        aux/=aux.sum()
        aux['woe'] = np.log(aux[0]/aux[1])
        aux['iv'] = (aux[0]-aux[1])*aux['woe']
        iv = aux['iv'].sum()
        return np.nan if np.isinf(iv) else iv 

    @staticmethod
    def _check_monotonic(X:pd.Series,y:pd.Series)->bool:
        """Validates if a given discretized feature has monotonic risk behavior 

        Args:
            X (pd.Series): Discretized predictors data
            y (pd.Series): Dichotomic response feature

        Returns:
            bool: Whether or not the feature has monotonic risk
        """
        aux = pd.concat([X,y],axis=1)
        aux.columns = ['x','y']
        aux = aux.loc[aux['x']!='MISSING'].reset_index(drop=True)
        aux = aux.groupby('x').mean()
        aux = list(aux['y'])
        return (len(aux)>=2) and ((sorted(aux) == aux)|(list(reversed(sorted(aux)))==aux))

class WoeContinuousFeatureSelector(WoeBaseFeatureSelector):
    """
        Class for selecting continuous features based on their WoE transformation and 
        Information Value statistic. 
    """
    selected_features = None 
    __is_fitted = False 
    _Xd = None 
    discretizers = None 

    def __init__(self) -> None:
        super().__init__()

    def fit(self,X:pd.DataFrame,y:pd.Series,method:str='quantile',iv_threshold:float=0.1,min_bins:int=2,max_bins:int=5,n_threads:int=1,strictly_monotonic:bool=False)->None:
        """Learns the best features given an IV threshold. Monotonic risk restriction can be applied. 

        Args:
            X (pd.DataFrame): Predictors data
            y (pd.Series): Dichotomic response feature
            method (str, optional): {'quantile','uniform','kmeans','gaussian','dcc','dec'}. Discretization 
            technique. For quantile, uniform, kmeans and gaussian only one method is applied. Regarding dcc and dce
            methods (Discrete Competitive Combination and Discrete Exhaustive Combination respectively), the following is performed:
            dcc: Selects the best discretization method for each predictor-
            dec: Includes the best feasible discretization for each method so they can complement each other.  
            Defaults to 'quantile'.
            iv_threshold (float, optional): IV value for a feature to be included in final selection. Defaults to 0.1.
            min_bins (int, optional): Minimun number of discretization bins. Defaults to 2.
            max_bins (int, optional): Maximun number of discretization bins. Defaults to 5.
            n_threads (int, optional): Number of multiprocessing threads. Defaults to 1.
            strictly_monotonic (bool, optional): Indicates if only monotonic risk features should be selected. 
            Defaults to False.

        Raises:
            Exception: If strictly_monotonic=True and no monotonic feature is present in the final selection.
        """
        
        cont_features = list(X.columns)
        if method in ('quantile','uniform','kmeans','gaussian'):
            disc = Discretizer(strategy=method,min_segments=min_bins,max_segments=max_bins)
            disc.fit(X[cont_features],n_threads=n_threads)
            self._Xd = disc.transform(X[cont_features])
            self._Xd['binary_target'] = y
            self.discretizers  =[disc]
        elif method in ('dcc','dec'):
            methods = ['quantile','uniform','kmeans','gaussian']
            discretizers = [Discretizer(strategy=method,min_segments=min_bins,max_segments=max_bins) for method in methods]
            for disc in discretizers:
                disc.fit(X[cont_features],n_threads=n_threads) 
            self.discretizers = discretizers[:]
            self._Xd = pd.concat([disc.transform(X[cont_features]) for disc in discretizers],axis=1)
            self._Xd['binary_target'] = y
        disc_features = list(self._Xd.columns)
        mono = None
        if strictly_monotonic:
            mono = dict([(feature,self._check_monotonic(self._Xd[feature],self._Xd['binary_target'])) for feature in disc_features])
            mono = {x:y for x,y in mono.items() if y}                
            if len(mono)==0:
                raise Exception('There is no monotonic feature.\n Please  try turning strictly_monotonic parameter to False or increase the number of bins')
            else:
                disc_features = [x for x,y in mono.items() if y]
        iv = [(feature,self._information_value(self._Xd[feature],self._Xd['binary_target'])) for feature in disc_features]
        iv = [(feature,value) for feature,value in iv if value>=iv_threshold]
        iv = pd.DataFrame(iv,columns=['feature','iv'])
        iv['root_feature'] = iv['feature'].map(lambda x:"_".join(x.split('_')[1:-2]))
        iv['nbins'] = iv['feature'].map(lambda x:x.split('_')[-2])
        iv['method'] = iv['feature'].map(lambda x:x.split('_')[-1])
        if method in ('quantile','uniform','kmeans','gaussian','dcc'):
            iv = iv.sort_values(by=['root_feature','iv','nbins'],ascending=[True,False,True]).reset_index(drop=True)
            iv['index'] = iv.groupby('root_feature').cumcount()+1
        elif method == 'dec':
            iv = iv.sort_values(by=['root_feature','method','iv','nbins'],ascending=[True,True,False,True]).reset_index(drop=True)
            iv['index'] = iv.groupby(['root_feature','method']).cumcount()+1
        iv = iv.loc[iv['index']==1].reset_index(drop=True)
        cont_features = list(set(iv['root_feature']))
        for disc in self.discretizers:
                disc.fit(X[cont_features],n_threads=n_threads) 
        self.selected_features =  iv.drop('index',axis=1).to_dict(orient='records')
        self.__is_fitted = True
    
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        """Converts continuous features to their best discretization

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
        else:
            aux = X.copy()
            features = list(set([feature['root_feature'] for feature in self.selected_features]))
            non_present_features = [f for f in features if f not in X.columns]
            if len(non_present_features)>0:
                print(f'{",".join(non_present_features)} feature{"s" if len(non_present_features)>1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                aux = pd.concat(map(lambda disc: disc.transform(X[features]),self.discretizers),axis=1)
                aux = aux[[feature['feature'] for feature in self.selected_features]]
                return aux

class WoeDiscreteFeatureSelector(WoeBaseFeatureSelector):
    """
        Class for selecting discrete features based on their WoE transformation and 
        Information Value statistic. 
    """
    def __init__(self):
        super().__init__()
    
    def fit(self,X:pd.DataFrame,y:pd.Series,iv_threshold:float=0.1)->None:
        """Learns best features given an IV threshold.

        Args:
            X (pd.DataFrame): Discrete predictors data
            y (pd.Series): Dichotomic response feature
            iv_threshold (float, optional):  IV value for a feature to be included in final selection. Defaults to 0.1.
        """
        disc_features = list(X.columns)
        aux = X.copy()
        aux['binary_target'] = y 
        iv = [(feature,self._information_value(aux[feature],aux['binary_target'])) for feature in disc_features]
        iv = [(feature,value) for feature,value in iv if value>=iv_threshold]
        iv = pd.DataFrame(iv,columns=['feature','iv'])
        disc_features = list(iv['feature'])
        self.selected_features =  iv.set_index('feature')['iv'].to_dict()
        self.__is_fitted = True        
    
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
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
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = [feature for feature in self.selected_features.keys()]
            non_present_features = [f for f in features if f not in X.columns]
            if len(non_present_features)>0:
                print(f'{",".join(non_present_features)} feature{"s" if len(non_present_features)>1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                aux = aux[features]
                return aux
        
class CreditScoring:
    """Implements credit risk scorecards following the methodology proposed in 
    Siddiqi, N. (2012). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
    """

    logistic_regression = None
    pdo = None
    base_odds = None
    base_score = None 
    betas = None 
    alpha = None 
    factor = None
    offset = None
    features = None 
    n = None
    scorecard = None 
    scoring_map = None 
    __is_fitted = False 

    def __init__(self,pdo:int=20,base_score:int=400,base_odds:int=1) -> None:
        """Initializes Credit Scoring object.

        Args:
            pdo (int, optional): Points to Double the Odd's _. Defaults to 20.
            base_score (int, optional): Default score for calibration. Defaults to 400.
            base_odds (int, optional): Odd's base at base_score . Defaults to 1.
        """
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = self.pdo/np.log(2)
        self.offset = self.base_score-self.factor*np.log(self.base_odds)
        
    @staticmethod
    def _get_scorecard(X:pd.DataFrame,feature:str)->pd.DataFrame:
        """Generates scorecard points for a given feature

        Args:
            X (pd.DataFrame): Feature Data
            feature (str): Predictor

        Returns:
            pd.DataFrame: Feature, Attribute and respective points
        """
        sc = X[[feature,f'P_{feature}']].copy().drop_duplicates().reset_index(drop=True)
        sc.rename(columns={feature:'attribute',f'P_{feature}':'points'},inplace=True)
        sc.insert(0,'feature',feature)
        return sc 

    def fit(self,Xw:pd.DataFrame,woe_encoder:WoeEncoder,logistic_regression:LogisticRegression)->None:
        """Learns scoring map

        Args:
            Xw (pd.DataFrame): WoE transformed data
            woe_encoder (WoeEncoder): WoE encoder fitted object
            logistic_regression (LogisticRegression): Fitted logistic regression model
        """
        X = Xw.copy()
        self.betas = list(logistic_regression.coef_[0])
        self.alpha = logistic_regression.intercept_[0]
        self.features = dict(zip(logistic_regression.feature_names_in_,self.betas))
        self.n = len(self.betas)
        for feature,beta in self.features.items():
            X[f'P_{feature}'] = np.floor((-X[feature]*beta+self.alpha/self.n)*self.factor+self.offset/self.n).astype(int)
        features = list(self.features.keys())
        X[features] = woe_encoder.inverse_transform(X[features])
        self.scorecard = pd.concat(map(lambda f:self._get_scorecard(X,f),features))
        self.scorecard = self.scorecard.groupby(['feature','attribute']).max()
        self.scoring_map = dict(ChainMap(*[{f:d[['attribute','points']].set_index('attribute')['points'].to_dict()} for f,d in self.scorecard.reset_index().groupby('feature')]))
        self.__is_fitted = True 
    
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
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
            raise Exception('Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = list(self.scoring_map.keys())
            non_present_features = [f for f in features if f not in aux.columns]
            if len(non_present_features)>0:
                print(f'{",".join(non_present_features)} feature{"s" if len(non_present_features)>1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                for feature,points_map in self.scoring_map.items():
                    aux[feature].replace(points_map,inplace=True)
                aux['score'] = aux[features].sum(axis=1) 
                return aux