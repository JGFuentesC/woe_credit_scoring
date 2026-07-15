
import pandas as pd
import numpy as np
import pytest
from CreditScoringToolkit import IVCalculator


@pytest.fixture
def data_continuous():
    np.random.seed(42)
    n = 100
    data = {
        'feat_a': np.random.normal(40, 10, n),
        'feat_b': np.random.exponential(50, n),
        'target': np.random.binomial(1, 0.3, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_discrete():
    data = {
        'cat_a': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'Z', 'X', 'Y', 'Z'],
        'cat_b': ['M', 'N', 'M', 'N', 'M', 'N', 'M', 'N', 'M', 'N'],
        'target': [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_both():
    np.random.seed(42)
    n = 100
    data = {
        'feat_a': np.random.normal(40, 10, n),
        'feat_b': np.random.exponential(50, n),
        'cat_a': np.random.choice(['X', 'Y', 'Z'], n),
        'cat_b': np.random.choice(['M', 'N'], n),
        'target': np.random.binomial(1, 0.3, n),
    }
    return pd.DataFrame(data)


def test_iv_calculator_continuous_only(data_continuous):
    calc = IVCalculator(
        data=data_continuous,
        target='target',
        continuous_features=['feat_a', 'feat_b'],
    )
    result = calc.calculate_iv(max_discretization_bins=5)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'feature', 'iv', 'feature_type'}
    assert len(result) == 2
    assert all(result['feature_type'] == 'continuous')
    assert all(isinstance(v, float) for v in result['iv'])


def test_iv_calculator_discrete_only(data_discrete):
    calc = IVCalculator(
        data=data_discrete,
        target='target',
        discrete_features=['cat_a', 'cat_b'],
    )
    result = calc.calculate_iv()

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'feature', 'iv', 'feature_type'}
    assert len(result) == 2
    assert all(result['feature_type'] == 'discrete')
    assert all(isinstance(v, float) for v in result['iv'])


def test_iv_calculator_both(data_both):
    calc = IVCalculator(
        data=data_both,
        target='target',
        continuous_features=['feat_a', 'feat_b'],
        discrete_features=['cat_a', 'cat_b'],
    )
    result = calc.calculate_iv(max_discretization_bins=5)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'feature', 'iv', 'feature_type'}
    assert len(result) == 4
    assert set(result['feature_type'].unique()) == {'continuous', 'discrete'}
    assert all(isinstance(v, float) for v in result['iv'])


def test_iv_calculator_type_error_not_dataframe():
    with pytest.raises(TypeError):
        IVCalculator(data=[1, 2, 3], target='target')


def test_iv_calculator_value_error_target_not_in_columns():
    df = pd.DataFrame({'col': [1, 2, 3]})
    with pytest.raises(ValueError):
        IVCalculator(data=df, target='missing_col')
