import pytest
import pandas as pd
import numpy as np
from CreditScoringToolkit import Discretizer


def test_fit_uniform_strategy(sample_continuous_df):
    disc = Discretizer(strategy='uniform')
    disc.fit(sample_continuous_df)
    assert disc._Discretizer__is_fitted
    assert len(disc.edges_map) > 0
    for entry in disc.edges_map:
        assert set(entry.keys()) == {'feature', 'nbins', 'edges'}


def test_fit_quantile_strategy(sample_continuous_df):
    disc = Discretizer(strategy='quantile')
    disc.fit(sample_continuous_df)
    assert disc._Discretizer__is_fitted
    assert len(disc.edges_map) > 0
    for entry in disc.edges_map:
        assert set(entry.keys()) == {'feature', 'nbins', 'edges'}


def test_fit_kmeans_strategy(sample_continuous_df):
    disc = Discretizer(strategy='kmeans')
    disc.fit(sample_continuous_df)
    assert disc._Discretizer__is_fitted
    assert len(disc.edges_map) > 0
    for entry in disc.edges_map:
        assert set(entry.keys()) == {'feature', 'nbins', 'edges'}


def test_fit_gaussian_strategy(sample_continuous_df):
    df = sample_continuous_df[['C_age', 'C_income', 'C_debt_ratio']].copy()
    disc = Discretizer(strategy='gaussian')
    disc.fit(df)
    assert disc._Discretizer__is_fitted
    assert len(disc.edges_map) > 0
    for entry in disc.edges_map:
        assert set(entry.keys()) == {'feature', 'nbins', 'edges'}
        assert len(entry['edges']) >= 3


def test_transform_returns_correct_shape(sample_continuous_df):
    min_segments, max_segments = 2, 5
    disc = Discretizer(min_segments=min_segments, max_segments=max_segments)
    disc.fit(sample_continuous_df)
    result = disc.transform(sample_continuous_df)
    n_features = len(sample_continuous_df.columns)
    n_nbins = max_segments - min_segments + 1
    expected_cols = n_features * n_nbins
    assert result.shape == (len(sample_continuous_df), expected_cols)


def test_transform_column_naming_convention(sample_continuous_df):
    disc = Discretizer(strategy='quantile', min_segments=2, max_segments=3)
    disc.fit(sample_continuous_df)
    result = disc.transform(sample_continuous_df)
    for col in result.columns:
        parts = col.split('_', 1)
        assert parts[0] == 'disc'
        remainder = parts[1]
        assert remainder.endswith('_quantile')
        inner = remainder[:-(len('_quantile'))]
        assert '_' in inner


def test_transform_nan_produces_missing_category(sample_continuous_df):
    disc = Discretizer(strategy='quantile')
    disc.fit(sample_continuous_df)
    result = disc.transform(sample_continuous_df)
    age_cols = [c for c in result.columns if 'C_age' in c]
    income_cols = [c for c in result.columns if 'C_income' in c]
    for col in age_cols:
        assert result.loc[0, col] == 'MISSING'
    for col in income_cols:
        assert result.loc[10, col] == 'MISSING'


def test_unfit_transform_raises(sample_continuous_df):
    disc = Discretizer()
    with pytest.raises(Exception, match='Please call fit method first'):
        disc.transform(sample_continuous_df)


def test_transform_missing_features_raises(sample_continuous_df):
    disc = Discretizer()
    disc.fit(sample_continuous_df[['C_age', 'C_income']])
    with pytest.raises(Exception, match='Missing features'):
        disc.transform(sample_continuous_df[['C_debt_ratio']])


def test_single_unique_value_column(sample_continuous_df):
    disc = Discretizer(strategy='quantile')
    df = sample_continuous_df[['C_constant']].copy()
    disc.fit(df)
    result = disc.transform(df)
    assert result.shape[0] == len(df)
    assert result.shape[1] > 0
    for entry in disc.edges_map:
        assert entry['edges'] == [-np.inf, np.inf]


def test_all_nan_column(sample_continuous_df):
    disc = Discretizer(strategy='quantile')
    df = sample_continuous_df[['C_full_nan']].copy()
    disc.fit(df)
    result = disc.transform(df)
    assert result.shape[0] == len(df)
    for col in result.columns:
        assert (result[col] == 'MISSING').all()


def test_empty_dataframe_returns_empty():
    disc = Discretizer()
    disc.fit(pd.DataFrame())
    result = disc.transform(pd.DataFrame(index=range(3)))
    assert result.shape == (3, 0)


def test_with_n_threads_greater_than_one(sample_continuous_df):
    disc_mt = Discretizer(strategy='quantile')
    disc_mt.fit(sample_continuous_df, n_threads=2)
    result_mt = disc_mt.transform(sample_continuous_df, n_threads=2)

    disc_st = Discretizer(strategy='quantile')
    disc_st.fit(sample_continuous_df, n_threads=1)
    result_st = disc_st.transform(sample_continuous_df, n_threads=1)

    pd.testing.assert_frame_equal(result_mt, result_st)
