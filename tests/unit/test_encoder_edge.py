
import pandas as pd
import numpy as np
import pytest
from CreditScoringToolkit import WoeEncoder


def test_encoder_raises_value_error_on_non_binary_target():
    df = pd.DataFrame({'feature': ['A', 'A', 'B', 'B', 'C']})
    target = pd.Series([0, 0, 1, 1, 2], name='target')
    encoder = WoeEncoder()
    with pytest.raises(ValueError, match="must have exactly 2 unique values"):
        encoder.fit(df, target)


def test_encoder_raises_value_error_on_single_value_target():
    df = pd.DataFrame({'feature': ['A', 'A', 'B']})
    target = pd.Series([1, 1, 1], name='target')
    encoder = WoeEncoder()
    with pytest.raises(ValueError, match="must have exactly 2 unique values"):
        encoder.fit(df, target)


def test_encoder_works_with_non_01_target():
    df = pd.DataFrame({'feature': ['A', 'A', 'A', 'B', 'B']})
    target = pd.Series([10, 10, 20, 20, 10], name='target')
    encoder = WoeEncoder()
    encoder.fit(df, target)
    transformed = encoder.transform(df)
    assert not transformed.isnull().values.any()
    assert transformed.shape == df.shape


def test_encoder_raises_on_unfitted_transform():
    encoder = WoeEncoder()
    df = pd.DataFrame({'feature': ['A', 'B']})
    with pytest.raises(Exception, match="Please call fit method first"):
        encoder.transform(df)


def test_encoder_raises_on_unfitted_inverse_transform():
    encoder = WoeEncoder()
    df = pd.DataFrame({'feature': [0.5, -0.2]})
    with pytest.raises(Exception, match="Please call fit method first"):
        encoder.inverse_transform(df)
