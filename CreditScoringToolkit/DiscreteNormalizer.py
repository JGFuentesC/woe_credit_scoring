import numpy as np
import pandas as pd
from collections import ChainMap
from itertools import repeat

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
        aux.columns = [feature]
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
                        aux[feat] = aux[feat].replace(dict(zip(new_categories,repeat(replacement))))
                    aux[feat] = aux[feat].replace(mapping)
                return aux 
