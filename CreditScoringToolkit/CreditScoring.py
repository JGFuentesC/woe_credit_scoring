import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from collections import ChainMap
from . import WoeEncoder 

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
        self.features = dict(zip(Xw.columns,self.betas))
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