
import pandas as pd
import numpy as np
from CreditScoringToolkit import WoeBaseFeatureSelector


def test_information_value_matches_expected():
    X = pd.Series(['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'], name='feat')
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 1, 1], name='target')

    iv = WoeBaseFeatureSelector._information_value(X, y)

    assert iv is not None
    assert np.isclose(iv, 0.5924585, rtol=1e-5)


def test_information_value_none_for_perfect_separation():
    X = pd.Series(['A', 'A', 'B', 'B'], name='feat')
    y = pd.Series([0, 0, 1, 1], name='target')

    iv = WoeBaseFeatureSelector._information_value(X, y)

    assert iv is None


def test_information_value_single_category():
    X = pd.Series(['A', 'A', 'A', 'A'], name='feat')
    y = pd.Series([0, 0, 1, 1], name='target')

    iv = WoeBaseFeatureSelector._information_value(X, y)

    assert iv is not None
    assert isinstance(iv, float)


def test_check_monotonic_returns_true():
    X = pd.Series(['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'], name='feat')
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 1, 1], name='target')

    assert WoeBaseFeatureSelector._check_monotonic(X, y) is True


def test_check_monotonic_returns_false():
    X = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'], name='feat')
    y = pd.Series([0, 1, 0, 0, 1, 1], name='target')

    assert WoeBaseFeatureSelector._check_monotonic(X, y) is False


def test_check_monotonic_excludes_missing():
    X = pd.Series(['A', 'A', 'A', 'MISSING', 'MISSING', 'B', 'B', 'B'], name='feat')
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1], name='target')

    assert WoeBaseFeatureSelector._check_monotonic(X, y) is True
