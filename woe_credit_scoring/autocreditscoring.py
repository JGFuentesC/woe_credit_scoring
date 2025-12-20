from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import ChainMap
import os
import logging
from .normalizer import DiscreteNormalizer
from .binning import WoeContinuousFeatureSelector, WoeDiscreteFeatureSelector
from .encoder import WoeEncoder
from .scoring import CreditScoring

logger = logging.getLogger("CreditScoringToolkit")

class AutoCreditScoring:
    """
    A class used to perform automated credit scoring using logistic regression and Weight of Evidence (WoE) transformation.
    Attributes
    ----------
    continuous_features : List[str]
        List of continuous feature names.
    discrete_features : List[str]
        List of discrete feature names.
    target : str
        The target variable name.
    data : pd.DataFrame
        The input data containing features and target.
    train : pd.DataFrame
        The training dataset.
    valid : pd.DataFrame
        The validation dataset.
    apply_multicolinearity : bool, optional
        Whether to apply multicollinearity treatment (default is False).
    iv_feature_threshold : float, optional
        The Information Value (IV) threshold for feature selection (default is 0.05).
    treat_outliers : bool, optional
        Whether to treat outliers in continuous features (default is False).
    outlier_threshold : float, optional
        The threshold for outlier treatment (default is 0.01).
    min_score : int, optional
        The minimum score for the credit scoring model (default is 400).
    max_score : int, optional
        The maximum score for the credit scoring model (default is 900).
    max_discretization_bins : int, optional
        The maximum number of bins for discretization (default is 5).
    discrete_normalization_threshold : float, optional
        The threshold for discrete feature normalization (default is 0.05).
    discrete_normalization_default_category : str, optional
        The default category for discrete feature normalization (default is 'OTHER').
    transformation : Optional[str], optional
        The transformation method to be applied (default is None).
    model : Optional[LogisticRegression], optional
        The logistic regression model (default is None).
    max_iter : int, optional
        The maximum number of iterations for partitioning data (default is 5).
    train_size : float, optional
        The proportion of data to be used for training (default is 0.7).
    target_proportion_tolerance : float, optional
        The tolerance for target proportion difference between train and valid datasets (default is 0.01).
    strictly_monotonic : bool, optional
        Whether to enforce strictly monotonic WoE transformation (default is True).
    discretization_method : str, optional
        The method for discretization (default is 'quantile').
    n_threads : int, optional
        The number of threads to use for parallel processing (default is 1).
    overfitting_tolerance : float, optional
        The tolerance for overfitting detection (default is 0.01).
    create_reporting : bool, optional
        Whether to create reporting after model fitting (default is False).
    is_fitted : bool, optional
        Whether the model has been fitted (default is False).
    Methods
    -------
    __init__(self, data: pd.DataFrame, target: str, continuous_features: List[str]=None, discrete_features: List[str]=None)
        Initializes the AutoCreditScoring object with data, target, and feature lists.
    fit(self, target_proportion_tolerance: float = None, treat_outliers: bool = None, discrete_normalization_threshold: float = None, discrete_normalization_default_category: str = None, max_discretization_bins: int = None, strictly_monotonic: bool = None, iv_feature_threshold: float = None, discretization_method: str = None, n_threads: int = None, overfitting_tolerance: float = None, min_score: int = None, max_score: int = None, create_reporting: bool = None, verbose: bool = False)
        Fits the credit scoring model to the data with optional parameters for customization.
    __partition_data(self)
        Partitions the data into training and validation sets while ensuring target proportion compatibility.
    __outlier_treatment(self)
        Applies outlier treatment to continuous features in the training dataset.
    __normalize_discrete(self)
        Normalizes discrete features in the training dataset.
    __feature_selection(self)
        Performs feature selection based on Information Value (IV) for continuous and discrete features.
    __woe_transformation(self)
        Applies Weight of Evidence (WoE) transformation to the selected features.
    __apply_pipeline(self, data: pd.DataFrame) -> pd.DataFrame
        Applies the entire preprocessing and transformation pipeline to new data.
    __train_model(self)
        Trains the logistic regression model on the transformed training data.
    __scoring(self)
        Generates credit scores for the training and validation datasets.
    __reporting(self)
        Creates various reports and visualizations for model evaluation and interpretation.
    save_reports(self, folder: str = '.')
        Saves the generated reports and visualizations to the specified folder.
    predict(self, X: pd.DataFrame) -> pd.DataFrame
        Predicts scores for a given raw dataset.
    fit_predict(self, **kwargs) -> pd.DataFrame
        Fits the model and returns the scores for the entire dataset.
    """
    continuous_features: List[str]
    discrete_features: List[str]
    target: str
    data: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    iv_feature_threshold: float = 0.05
    treat_outliers: bool = False
    outlier_threshold: float = 0.01
    min_score = 400
    max_score = 900
    max_discretization_bins = 5
    discrete_normalization_threshold = 0.05
    discrete_normalization_default_category = 'OTHER'
    transformation: Optional[str] = None
    model: Optional[LogisticRegression] = None
    max_iter: int = 5
    train_size: float = 0.7
    target_proportion_tolerance: float = 0.01
    max_discretization_bins:int=6
    strictly_monotonic:bool=True
    discretization_method:str = 'quantile'
    n_threads:int = 1
    overfitting_tolerance:float = 0.01
    create_reporting:bool = False
    is_fitted:bool = False

    def __init__(self, data: pd.DataFrame, target: str, continuous_features: List[str]=None, discrete_features: List[str]=None):
        self.data = data
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        self.target = target

    def fit(self,
            target_proportion_tolerance:float = None,
            train_proportion:float = None,
            treat_outliers:bool = None,
            discrete_normalization_threshold:float = None,
            discrete_normalization_default_category:str = None,
            max_discretization_bins:int = None,
            strictly_monotonic:bool = None,
            iv_feature_threshold:float = None,
            discretization_method:str = None,
            n_threads:int = None,
            overfitting_tolerance:float = None,
            min_score:int = None,
            max_score:int = None,
            create_reporting:bool = None,
            verbose:bool=False):

        #Train proportion control
        if train_proportion is not None:
            self.train_size = train_proportion

        # Verbosity control
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        # Check if continuous_features is provided
        if self.continuous_features is None:
            self.continuous_features = []
            logger.warning("No continuous features provided")
        # Check if discrete_features is provided
        if self.discrete_features is None:
            self.discrete_features = []
            logger.warning("No discrete features provided")
        if len(self.continuous_features)==0 and len(self.discrete_features)==0:
            logger.error("No features provided")
            raise RuntimeError("No features provided")

        # Check if target_proportion_tolerance is provided
        if target_proportion_tolerance is not None:
            self.target_proportion_tolerance = target_proportion_tolerance
        # Partition data
        self.__partition_data()

        #Check if treat_outliers is provided
        if len(self.continuous_features)>0 and treat_outliers is not None:
            self.treat_outliers = treat_outliers
            self.__outlier_treatment()

        # Check if discrete_normalization_threshold is provided
        if discrete_normalization_threshold is not None:
            self.discrete_normalization_threshold = discrete_normalization_threshold
        # Check if discrete_normalization_default_category is provided
        if discrete_normalization_default_category is not None:
            self.discrete_normalization_default_category = discrete_normalization_default_category
        if len(self.discrete_features)==0:
            logger.warning("No discrete features provided")
        else:
            if len(self.discrete_features)>0:
                # Normalize discrete features
                self.__normalize_discrete()

        #Check feature selection parameters
        if max_discretization_bins is not None:
            self.max_discretization_bins = max_discretization_bins
        if strictly_monotonic is not None:
            self.strictly_monotonic = strictly_monotonic
        if iv_feature_threshold is not None:
            self.iv_feature_threshold = iv_feature_threshold
        if discretization_method is not None:
            self.discretization_method = discretization_method
        if n_threads is not None:
            self.n_threads = n_threads

        # Feature selection
        self.__feature_selection()

        # Woe transformation
        self.__woe_transformation()

        # Check if overfitting_tolerance is provided
        if overfitting_tolerance is not None:
            self.overfitting_tolerance = overfitting_tolerance
        # Train model
        self.__train_model()

        # Check if min_score is provided
        if min_score is not None:
            self.min_score = min_score
        # Check if max_score is provided
        if max_score is not None :
            self.max_score = max_score
        # Check if min_score is less than max_score
        if self.min_score>=self.max_score:
            logger.error("min_score should be less than max_score")
            raise RuntimeError("min_score should be less than max_score")
        # Scoring
        self.__scoring()

        # Check if create_reporting is provided
        if create_reporting is not None:
            self.create_reporting = create_reporting
        # Reporting
        if self.create_reporting:
            self.__reporting()
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts scores for a given raw dataset.

        The input data should have the same features as the training data.
        The method applies the same pipeline of transformations as used during training.

        Args:
            X (pd.DataFrame): Raw data to be scored.

        Returns:
            pd.DataFrame: A DataFrame with scores and feature contributions.

        Raises:
            Exception: If the model is not fitted yet.
            ValueError: If the input data is missing required features.
        """
        if not self.is_fitted:
            raise Exception("This AutoCreditScoring instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        required_features = self.continuous_features + self.discrete_features
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"The following required columns are missing from the input data: {', '.join(missing_features)}")

        aux = X.copy()

        # Apply the full pipeline
        data_woe = self.__apply_pipeline(aux)

        # Inverse transform to get discrete bins
        data_discrete_binned = self.woe_encoder.inverse_transform(data_woe)

        # Get scores
        scored_data = self.credit_scoring.transform(data_discrete_binned)

        rename_dict = {
            binned_name: f"pts_{original_name}"
            for binned_name, original_name in self.feature_name_mapping.items()
            if binned_name in scored_data.columns
        }
        scored_data.rename(columns=rename_dict, inplace=True)

        scored_data['score'] = scored_data['score'].astype(float)

        # Clip scores to the defined range
        scored_data['score'] = scored_data['score'].clip(self.min_score, self.max_score)

        # Add score ranges
        for k in [5, 10]:
            step = (self.max_score - self.min_score) / k
            bins = np.arange(self.min_score, self.max_score + step, step)
            scored_data[f'range_score_{k}'] = pd.cut(scored_data['score'], bins=bins, include_lowest=True)

        return scored_data

    def __partition_data(self):
        logger.info("Partitioning data...")
        self.train, self.valid = train_test_split(self.data, train_size=self.train_size)
        self.train.reset_index(drop=True, inplace=True)
        self.valid.reset_index(drop=True, inplace=True)
        # Check if target proportions are compatible between train and valid
        logger.info("Checking partition proportions...")
        iter = 1
        while(np.abs(self.train[self.target].mean()-self.valid[self.target].mean())>self.target_proportion_tolerance):
            logger.info(f"Partitioning data...Iteration {iter}")
            logger.info(f"Train target proportion: {self.train[self.target].mean()}")
            logger.info(f"Valid target proportion: {self.valid[self.target].mean()}")
            self.train, self.valid = train_test_split(self.data, train_size=self.train_size)
            self.train.reset_index(drop=True, inplace=True)
            self.valid.reset_index(drop=True, inplace=True)
            iter+=1
            if iter>self.max_iter:
                logger.error("Could not find a compatible partition")
                raise RuntimeError("Could not find a compatible partition")

        if iter>1:
            logger.info(f"Partitioning data...Done after {iter} iterations")
        logger.info(f"Train shape: {self.train.shape}", )
        logger.info(f"Test shape: {self.valid.shape}")
        logger.info(f"Train target proportion: {self.train[self.target].mean()}")
        logger.info(f"Valid target proportion: {self.valid[self.target].mean()}")

    def __outlier_treatment(self):
        logger.info("Outlier treatment...")
        before = self.train[self.continuous_features].mean()
        for f in self.continuous_features:
            self.train[f] = winsorize(self.train[f], limits=[self.outlier_threshold, self.outlier_threshold])
        after = self.train[self.continuous_features].mean()
        report = pd.DataFrame({'Before':before,'After':after})
        logger.info("Mean statistics before and after outlier treatment")
        logger.info(f'\n\n{report}\n')
        logger.info("Outlier treatment...Done")

    def __normalize_discrete(self):
        logger.info("Discrete normalization...")
        logger.info(f"Discrete features: {self.discrete_features}")
        dn = DiscreteNormalizer(normalization_threshold=self.discrete_normalization_threshold,
                                default_category=self.discrete_normalization_default_category)
        dn.fit(self.train[self.discrete_features])
        self.train_discrete_normalized = dn.transform(self.train[self.discrete_features])
        logger.info("Checking if normalization produced unary columns")
        self.unary_columns = [c for c in self.train_discrete_normalized.columns if self.train_discrete_normalized[c].nunique()==1]
        if len(self.unary_columns)>0:
            logger.warning(f"Normalization produced unary columns: {self.unary_columns}")
            logger.warning(f"Removing unary columns from discrete features")
            self.discrete_features = [f for f in self.discrete_features if f not in self.unary_columns]
            logger.warning(f"Discrete features after unary columns removal: {self.discrete_features}")
        else:
            logger.info("No unary columns produced by normalization")
        if len(self.discrete_features)==0:
            logger.warning("No discrete features left after normalization")
        else:
            dn.fit(self.train[self.discrete_features])
            self.train_discrete_normalized = dn.transform(self.train[self.discrete_features])
        self.discrete_normalizer = dn
        logger.info("Discrete normalization...Done")

    def __feature_selection(self):
        try:
            logger.info("Feature selection...")
            if len(self.continuous_features)>0:
                logger.info("Continuous features selection...")
                woe_continuous_selector = WoeContinuousFeatureSelector()
                woe_continuous_selector.fit(self.train[self.continuous_features], self.train[self.target],
                    max_bins=self.max_discretization_bins,
                    strictly_monotonic=self.strictly_monotonic,
                    iv_threshold=self.iv_feature_threshold,
                    method=self.discretization_method,
                    n_threads=self.n_threads)
                self.iv_report_continuous = pd.DataFrame(woe_continuous_selector.selected_features)
                self.full_iv_report_continuous = woe_continuous_selector.iv_report.copy()
                self.continuous_candidate = woe_continuous_selector.transform(self.train[self.continuous_features])
                logger.info(f'\n\n{self.iv_report_continuous}\n\n')
                self.woe_continuous_selector = woe_continuous_selector
                logger.info(f"Continuous features selection...Done")
            if len(self.discrete_features)>0:
                logger.info("Discrete features selection...")
                woe_discrete_selector = WoeDiscreteFeatureSelector()
                woe_discrete_selector.fit(self.train_discrete_normalized, self.train[self.target],self.iv_feature_threshold)
                self.iv_report_discrete = pd.Series(woe_discrete_selector.selected_features).to_frame('iv').reset_index().rename(columns={'index':'feature'}).sort_values('iv',ascending=False)
                self.full_iv_report_discrete = woe_discrete_selector.iv_report.copy()
                self.discrete_candidate = woe_discrete_selector.transform(self.train_discrete_normalized)
                logger.info(f'\n\n{self.iv_report_discrete}\n\n')
                self.woe_discrete_selector = woe_discrete_selector
                logger.info("Discrete features selection...Done")

            if len(self.continuous_features)>0 and len(self.discrete_features)>0:
                logger.info("Merging continuous and discrete features...")
                self.train_candidate = pd.concat([self.continuous_candidate, self.discrete_candidate], axis=1)
                logger.info("Merging continuous and discrete features...Done")
            elif len(self.continuous_features)>0:
                self.train_candidate = self.continuous_candidate
            elif len(self.discrete_features)>0:
                self.train_candidate = self.discrete_candidate
            self.candidate_features = list(self.train_candidate.columns)
            if len(self.candidate_features)==0:
                logger.error("No features selected")
                raise RuntimeError("No features selected")
            logger.info(f"Selected features ({len(self.candidate_features)}): {self.candidate_features}")

            self.feature_name_mapping = {}
            if len(self.continuous_features)>0 and hasattr(self, 'iv_report_continuous'):
                self.feature_name_mapping.update(self.iv_report_continuous.set_index('feature')['root_feature'].to_dict())

            if len(self.discrete_features)>0 and hasattr(self, 'woe_discrete_selector') and self.woe_discrete_selector.selected_features:
                self.feature_name_mapping.update({f: f for f in self.woe_discrete_selector.selected_features.keys()})

            logger.info("Feature selection...Done")
        except Exception as err:
            logger.error(f"Error in feature selection: {err}")
            raise err

    def __woe_transformation(self):
        self.woe_encoder = WoeEncoder()
        self.woe_encoder.fit(self.train_candidate, self.train[self.target])
        self.train_woe = self.woe_encoder.transform(self.train_candidate)
        if self.train_woe.isna().max().max():
            logger.error("NAs found in transformed data")
            raise RuntimeError("NAs found in transformed data, Maybe tiny missing in continuous?")

    def __apply_pipeline(self,data:pd.DataFrame)->pd.DataFrame:
        try:
            if len(self.continuous_features)>0:
                if self.treat_outliers:
                    for f in self.continuous_features:
                        data[f] = winsorize(data[f], limits=[self.outlier_threshold, self.outlier_threshold])
                data_continuous_candidate = self.woe_continuous_selector.transform(data[self.continuous_features])
            if len(self.discrete_features)>0:
                data_discrete_normalized = self.discrete_normalizer.transform(data[self.discrete_features])
                data_discrete_candidate = self.woe_discrete_selector.transform(data_discrete_normalized)
            if len(self.continuous_features)>0 and len(self.discrete_features)==0:
                data_candidate = data_continuous_candidate.copy()
            if len(self.continuous_features)==0 and len(self.discrete_features)>0:
                data_candidate = data_discrete_candidate.copy()
            if len(self.continuous_features)>0 and len(self.discrete_features)>0:
                data_candidate = pd.concat([data_continuous_candidate, data_discrete_candidate], axis=1)
            data_woe = self.woe_encoder.transform(data_candidate)
            if data_woe.isna().max().max():
                logger.error("NAs found in transformed data")
                raise RuntimeError("NAs found in transformed data, Maybe tiny missing in continuous?")
            return data_woe
        except Exception as err:
            logger.error(f"Error applying pipeline: {err}")
            raise err

    def __train_model(self):
        logger.info("Training model...")
        lr = LogisticRegression()
        lr.fit(self.train_woe,self.train[self.target])
        self.model = lr
        self.valid_woe = self.__apply_pipeline(self.valid)
        self.auc_train = roc_auc_score(y_score=lr.predict_proba(self.train_woe)[:,1],y_true=self.train[self.target])
        self.auc_valid = roc_auc_score(y_score=lr.predict_proba(self.valid_woe)[:,1],y_true=self.valid[self.target])
        logger.info(f"AUC for training: {self.auc_train}")
        logger.info(f"AUC for validation:{self.auc_valid}")
        self.betas = lr.coef_[0]
        self.alpha = lr.intercept_[0]
        if any([np.abs(b)<0.0001 for b in self.betas]):
            logger.warning("Some betas are close to zero, consider removing features")
            logger.warning(f"Betas: {dict(zip(self.candidate_features,self.betas))}")
            logger.warning(f"Suspicious features: {[f for f,b in zip(self.candidate_features,self.betas) if np.abs(b)<0.0001]}")
        if abs(self.auc_train-self.auc_valid)>self.overfitting_tolerance:
            logger.warning(f"Overfitting detected, review your hyperparameters. train_auc: {self.auc_train}, valid_auc: {self.auc_valid}")
        self.logistic_model = lr
        logger.info("Training model...Done")

    def __scoring(self):
        logger.info("Scoring...")
        cs = CreditScoring()
        cs.fit(self.train_woe, self.woe_encoder, self.logistic_model)
        self.credit_scoring = cs

        # Get original scores to find min/max for scaling
        scored_train_orig = self.credit_scoring.transform(self.woe_encoder.inverse_transform(self.train_woe))
        scored_valid_orig = self.credit_scoring.transform(self.woe_encoder.inverse_transform(self.valid_woe))

        self.min_output_score = min(scored_train_orig['score'].min(), scored_valid_orig['score'].min())
        self.max_output_score = max(scored_train_orig['score'].max(), scored_valid_orig['score'].max())

        logger.info(f"Min output score: {self.min_output_score}")
        logger.info(f"Max output score: {self.max_output_score}")
        logger.info(f"Linear transformation to a {self.min_score}-{self.max_score} scale")

        n = self.credit_scoring.n

        if self.max_output_score == self.min_output_score:
            logger.warning("All scores are the same, cannot apply linear transformation. Setting all scores to the average of min_score and max_score.")
            avg_score = (self.min_score + self.max_score) / 2
            self.credit_scoring.scorecard['points'] = np.floor(avg_score / n).astype(int)
        else:
            # Scaling parameters
            a = (self.max_score - self.min_score) / (self.max_output_score - self.min_output_score)
            b = self.min_score - a * self.min_output_score
            # Update scorecard points
            self.credit_scoring.scorecard['points'] = np.floor(a * self.credit_scoring.scorecard['points'] + b / n).astype(int)

        # Update scoring_map from the updated scorecard
        self.credit_scoring.scoring_map = dict(ChainMap(*[{f: d[['attribute', 'points']].set_index('attribute')['points'].to_dict()} for f, d in self.credit_scoring.scorecard.reset_index().groupby('feature')]))

        # Recalculate scores with the updated scorecard
        self.scored_train = self.credit_scoring.transform(self.woe_encoder.inverse_transform(self.train_woe))
        self.scored_valid = self.credit_scoring.transform(self.woe_encoder.inverse_transform(self.valid_woe))

        self.scored_train['score'] = self.scored_train['score'].astype(float)
        self.scored_valid['score'] = self.scored_valid['score'].astype(float)

        self.scored_train['score'] = self.scored_train['score'].clip(self.min_score, self.max_score)
        self.scored_valid['score'] = self.scored_valid['score'].clip(self.min_score, self.max_score)

        logger.info(f'Transformed min score: {self.scored_train["score"].min()}')
        logger.info(f'Transformed max score: {self.scored_train["score"].max()}')

        for k in [5,10]:
            step = (self.max_score-self.min_score)/k
            bins = np.arange(self.min_score, self.max_score+step, step)
            self.scored_train[f'range_score_{k}'] = pd.cut(self.scored_train['score'],bins=bins,include_lowest=True)
            self.scored_valid[f'range_score_{k}'] = pd.cut(self.scored_valid['score'],bins=bins,include_lowest=True)
        logger.info("Scoring...Done")

    def __reporting(self):
        logger.info("Reporting...")
        # Distribution images
        logger.info("Score Distribution images...")
        fig, ax = plt.subplots()
        sns.histplot(self.scored_train['score'], kde=False, stat='density', ax=ax, label='Train')
        sns.histplot(self.scored_valid['score'], kde=False, stat='density', ax=ax, label='Valid')
        ax.set_title("Score histogram")
        ax.legend()
        self.score_histogram_fig = fig

        fig, ax = plt.subplots()
        sns.kdeplot(self.scored_train['score'], ax=ax, label='Train')
        sns.kdeplot(self.scored_valid['score'], ax=ax, label='Valid')
        ax.set_title("Score KDE")
        ax.legend()
        self.score_kde_fig = fig
        # Event rate images
        logger.info("Event rate images...")
        self.event_rate_figs = []
        for k in [5,10]:
            fig, ax = plt.subplots()
            ax = pd.crosstab(self.scored_train[f'range_score_{k}'], self.train[self.target], normalize='index').plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Event rate by score range ({k} bins)")
            setattr(self,f'event_rate_fig_{k}',fig)
            self.event_rate_figs.append(fig)
        # IV report
        logger.info("IV report...")
        iv_reports = []
        if hasattr(self, 'iv_report_continuous'):
            iv_reports.append(self.iv_report_continuous[['root_feature','iv']].rename(columns={'root_feature':'feature'}))
        if hasattr(self, 'iv_report_discrete'):
            iv_reports.append(self.iv_report_discrete[['feature','iv']])

        if iv_reports:
            self.iv_report = pd.concat(iv_reports, axis=0).sort_values('iv', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(data=self.iv_report, x='iv', y='feature', ax=ax)
            ax.set_title("IV report")
        else:
            self.iv_report = pd.DataFrame(columns=['feature', 'iv'])
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No IV report generated.", horizontalalignment='center', verticalalignment='center')
            ax.set_title("IV report")
        self.iv_report_fig = fig
        # ROC Curve
        logger.info("ROC Curve...")
        fpr_train, tpr_train, _ = roc_curve(self.train[self.target], self.model.predict_proba(self.train_woe)[:, 1])
        fpr_valid, tpr_valid, _ = roc_curve(self.valid[self.target], self.model.predict_proba(self.valid_woe)[:, 1])
        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_valid = auc(fpr_valid, tpr_valid)
        fig, ax = plt.subplots()
        ax.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
        ax.plot(fpr_valid, tpr_valid, color='red', lw=2, label=f'Valid ROC curve (area = {roc_auc_valid:.2f})')
        ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        self.roc_curve_fig = fig
        logger.info("ROC Curve...Done")

    def save_reports(self,folder='.'):
        if not self.create_reporting:
            raise RuntimeError("Reports were not generated. Please run fit() with create_reporting=True before saving reports.")

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.score_histogram_fig.savefig(f'{folder}/score_histogram.png')
        self.score_kde_fig.savefig(f'{folder}/score_kde.png')
        self.iv_report_fig.savefig(f'{folder}/iv_report.png')
        for k in [5,10]:
            getattr(self,f'event_rate_fig_{k}').savefig(f'{folder}/event_rate_{k}.png')
        self.roc_curve_fig.savefig(f'{folder}/roc_curve.png')
        logger.info(f"Reports saved in {folder}")
