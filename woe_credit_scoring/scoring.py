from typing import Optional, Dict
from collections import ChainMap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
from .encoder import WoeEncoder

logger = logging.getLogger("CreditScoringToolkit")

class CreditScoring:
    """
    Implements credit risk scorecards following the methodology proposed in
    Siddiqi, N. (2012). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.

    This class provides methods to fit a logistic regression model to the provided data,
    transform the data using Weight of Evidence (WoE) encoding, and generate a scorecard
    that maps the model's coefficients to a scoring system. The scorecard can then be used
    to convert new data into credit scores.

    Attributes:
        logistic_regression (Optional[LogisticRegression]): Fitted logistic regression model.
        pdo (Optional[int]): Points to Double the Odds.
        base_odds (Optional[int]): Base odds at the base score.
        base_score (Optional[int]): Base score for calibration.
        betas (Optional[list]): Coefficients of the logistic regression model.
        alpha (Optional[float]): Intercept of the logistic regression model.
        factor (Optional[float]): Factor used in score calculation.
        offset (Optional[float]): Offset used in score calculation.
        features (Optional[Dict[str, float]]): Mapping of feature names to their coefficients.
        n (Optional[int]): Number of features.
        scorecard (Optional[pd.DataFrame]): DataFrame containing the scorecard.
        scoring_map (Optional[Dict[str, Dict[str, int]]]): Mapping of features to their score mappings.
        __is_fitted (bool): Indicates whether the model has been fitted.
    """

    logistic_regression: Optional[LogisticRegression] = None
    pdo: Optional[int] = None
    base_odds: Optional[int] = None
    base_score: Optional[int] = None
    betas: Optional[list] = None
    alpha: Optional[float] = None
    factor: Optional[float] = None
    offset: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    n: Optional[int] = None
    scorecard: Optional[pd.DataFrame] = None
    scoring_map: Optional[Dict[str, Dict[str, int]]] = None
    __is_fitted: bool = False

    def __init__(self, pdo: int = 20, base_score: int = 400, base_odds: int = 1) -> None:
        """Initializes Credit Scoring object.

        Args:
            pdo (int, optional): Points to Double the Odd's _. Defaults to 20.
            base_score (int, optional): Default score for calibration. Defaults to 400.
            base_odds (int, optional): Odd's base at base_score . Defaults to 1.
        """
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)

    @staticmethod
    def _get_scorecard(X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Generates scorecard points for a given feature

        Args:
            X (pd.DataFrame): Feature Data
            feature (str): Predictor

        Returns:
            pd.DataFrame: Feature, Attribute and respective points
        """
        sc = X[[feature, f'P_{feature}']].copy(
        ).drop_duplicates().reset_index(drop=True)
        sc = sc.rename(columns={feature: 'attribute',
                  f'P_{feature}': 'points'})
        sc.insert(0, 'feature', feature)
        return sc

    def fit(self, Xw: pd.DataFrame, woe_encoder: WoeEncoder, logistic_regression: LogisticRegression) -> None:
        """Learns scoring map

        Args:
            Xw (pd.DataFrame): WoE transformed data
            woe_encoder (WoeEncoder): WoE encoder fitted object
            logistic_regression (LogisticRegression): Fitted logistic regression model
        """
        X = Xw.copy()
        self.betas = list(logistic_regression.coef_[0])
        self.alpha = logistic_regression.intercept_[0]
        self.features = dict(zip(Xw.columns, self.betas))
        self.n = len(self.betas)
        for feature, beta in self.features.items():
            X[f'P_{feature}'] = np.floor(
                (-X[feature] * beta + self.alpha / self.n) * self.factor + self.offset / self.n).astype(int)
        features = list(self.features.keys())
        X[features] = woe_encoder.inverse_transform(X[features])
        self.scorecard = pd.concat(
            map(lambda f: self._get_scorecard(X, f), features))
        self.scorecard = self.scorecard.groupby(['feature', 'attribute']).max()
        self.scoring_map = dict(ChainMap(*[{f: d[['attribute', 'points']].set_index('attribute')[
                                'points'].to_dict()} for f, d in self.scorecard.reset_index().groupby('feature')]))
        self.__is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
            raise Exception(
                'Please call fit method first with the required parameters')
        else:
            aux = X.copy()
            features = list(self.scoring_map.keys())
            non_present_features = [
                f for f in features if f not in aux.columns]
            if len(non_present_features) > 0:
                logger.exception(
                    f'{",".join(non_present_features)} feature{"s" if len(non_present_features) > 1 else ""} not present in data')
                raise Exception("Missing features")
            else:
                for feature, points_map in self.scoring_map.items():
                    aux[feature] = aux[feature].replace(points_map)
                aux['score'] = aux[features].sum(axis=1)
                return aux
