import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from CreditScoringToolkit import (
    WoeEncoder, DiscreteNormalizer, Discretizer,
    WoeContinuousFeatureSelector, WoeDiscreteFeatureSelector,
    CreditScoring, IVCalculator, AutoCreditScoring
)


@pytest.fixture
def sample_discrete_df():
    data = {
        'cat_A': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'Z', 'X', 'Y', 'Z'],
        'cat_B': ['M', 'N', 'M', 'N', 'M', 'N', 'M', 'N', 'M', 'N'],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_target():
    return pd.Series([0, 0, 0, 1, 1, 1, 1, 0, 0, 1], name='target')


@pytest.fixture
def sample_continuous_df():
    np.random.seed(42)
    n = 100
    data = {
        'C_age': np.random.normal(40, 10, n),
        'C_income': np.random.exponential(50000, n),
        'C_debt_ratio': np.random.beta(2, 5, n),
        'C_constant': np.ones(n) * 5,
        'C_full_nan': [np.nan] * n,
    }
    df = pd.DataFrame(data)
    df.loc[0, 'C_age'] = np.nan
    df.loc[10, 'C_income'] = np.nan
    return df


@pytest.fixture
def sample_binary_target():
    np.random.seed(42)
    n = 100
    return pd.Series(np.random.binomial(1, 0.3, n), name='TARGET')


@pytest.fixture
def fitted_encoder(sample_discrete_df, sample_target):
    encoder = WoeEncoder()
    encoder.fit(sample_discrete_df, sample_target)
    return encoder


@pytest.fixture
def fitted_normalizer(sample_discrete_df):
    dn = DiscreteNormalizer(normalization_threshold=0.1)
    dn.fit(sample_discrete_df)
    return dn


@pytest.fixture
def train_data():
    data = pd.read_csv('example_data/train.csv')
    return data


@pytest.fixture
def valid_data():
    data = pd.read_csv('example_data/valid.csv')
    return data
