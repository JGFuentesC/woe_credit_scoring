import numpy as np
import pandas as pd
from collections import ChainMap

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
