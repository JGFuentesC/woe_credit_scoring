import numpy as np
import pandas as pd

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
