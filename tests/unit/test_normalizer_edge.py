
import pandas as pd
import numpy as np
import pytest
from CreditScoringToolkit import DiscreteNormalizer


def test_normalizer_threshold_zero_nothing_grouped():
    df = pd.DataFrame({
        'feature': ['A', 'A', 'B', 'B', 'B', 'C', 'D'],
    })
    dn = DiscreteNormalizer(normalization_threshold=0)
    dn.fit(df)
    transformed = dn.transform(df)
    assert 'OTHER' not in transformed['feature'].unique()
    assert set(transformed['feature'].unique()) == {'A', 'B', 'C', 'D'}


def test_normalizer_threshold_over_one_everything_grouped():
    df = pd.DataFrame({
        'feature': ['A', 'A', 'B', 'B', 'B', 'C', 'D'],
    })
    dn = DiscreteNormalizer(normalization_threshold=1.5, default_category='GROUPED')
    dn.fit(df)
    transformed = dn.transform(df)
    assert len(transformed['feature'].unique()) == 1


def test_normalizer_empty_dataframe_no_columns():
    df = pd.DataFrame()
    dn = DiscreteNormalizer()
    dn.fit(df)
    transformed = dn.transform(df)
    assert transformed.empty
    assert list(transformed.columns) == []


def test_normalizer_raises_type_error_on_non_dataframe():
    dn = DiscreteNormalizer()
    with pytest.raises(TypeError, match="Please use a Pandas DataFrame object"):
        dn.fit([1, 2, 3])
