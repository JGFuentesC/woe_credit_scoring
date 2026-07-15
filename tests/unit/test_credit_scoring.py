import logging
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from CreditScoringToolkit import WoeEncoder, CreditScoring


@pytest.fixture
def scoring_data():
    """Fixture with 2 features, 2 categories each, every cat has both classes."""
    df = pd.DataFrame({
        'feat_1': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B',
                   'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        'feat_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y',
                   'Y', 'Y', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y'],
        'target': [0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
                   1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    })
    data = df[['feat_1', 'feat_2']].copy()
    target = pd.Series(df['target'], name='target')
    return data, target


@pytest.fixture
def fitted_scoring(scoring_data):
    """Fit encoder, logistic regression, and CreditScoring; return cs and data."""
    data, target = scoring_data

    woe_encoder = WoeEncoder()
    woe_encoder.fit(data, target)
    Xw = woe_encoder.transform(data)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xw, target)

    cs = CreditScoring()
    cs.fit(Xw, woe_encoder, lr)

    return cs, data


def test_normal_scoring_produces_valid_scorecard(fitted_scoring):
    cs, data = fitted_scoring

    assert cs.scorecard is not None
    assert isinstance(cs.scorecard, pd.DataFrame)
    assert not cs.scorecard.empty
    assert 'points' in cs.scorecard.columns
    assert len(cs.scorecard) == data.nunique().sum()


def test_transform_produces_score_column(fitted_scoring):
    cs, data = fitted_scoring

    result = cs.transform(data)

    assert 'score' in result.columns
    assert pd.api.types.is_numeric_dtype(result['score'])
    assert len(result) == len(data)


def test_unfitted_transform_raises_exception():
    cs = CreditScoring()
    dummy = pd.DataFrame({'feat_1': ['A'], 'feat_2': ['X']})

    with pytest.raises(Exception, match='call fit method first'):
        cs.transform(dummy)


def test_transform_with_missing_features_raises_exception(fitted_scoring):
    cs, data = fitted_scoring
    incomplete = data[['feat_1']]

    with pytest.raises(Exception, match='Missing features'):
        cs.transform(incomplete)


def test_transform_with_unseen_categories_does_not_crash(fitted_scoring):
    cs, data = fitted_scoring
    new_data = data.head(3).copy()
    new_data['feat_1'] = 'unseen'

    result = cs.transform(new_data)

    assert 'score' in result.columns


def test_transform_with_unseen_categories_warns(fitted_scoring, caplog):
    cs, data = fitted_scoring
    new_data = data.head(3).copy()
    new_data['feat_1'] = 'unseen'

    with caplog.at_level(logging.WARNING):
        cs.transform(new_data)

    assert 'Unseen categories' in caplog.text


def test_pdo_zero_edge_case(scoring_data):
    data, target = scoring_data

    woe_encoder = WoeEncoder()
    woe_encoder.fit(data, target)
    Xw = woe_encoder.transform(data)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xw, target)

    cs = CreditScoring(pdo=0)
    assert cs.factor == 0.0
    assert cs.offset == 400.0

    cs.fit(Xw, woe_encoder, lr)

    assert cs.scorecard is not None
    assert not cs.scorecard.empty


def test_base_odds_zero_sets_offset_to_infinity():
    cs = CreditScoring(base_odds=0)

    assert cs.factor > 0
    assert np.isinf(cs.offset)
    assert cs.offset > 0


def test_custom_pdo_base_score_base_odds(scoring_data):
    data, target = scoring_data

    woe_encoder = WoeEncoder()
    woe_encoder.fit(data, target)
    Xw = woe_encoder.transform(data)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xw, target)

    cs = CreditScoring(pdo=30, base_score=500, base_odds=5)
    assert cs.pdo == 30
    assert cs.base_score == 500
    assert cs.base_odds == 5
    expected_factor = 30 / np.log(2)
    expected_offset = 500 - expected_factor * np.log(5)
    assert np.isclose(cs.factor, expected_factor)
    assert np.isclose(cs.offset, expected_offset)

    cs.fit(Xw, woe_encoder, lr)

    result = cs.transform(data)
    assert 'score' in result.columns


def test_scorecard_has_expected_columns(fitted_scoring):
    cs, data = fitted_scoring

    sc = cs.scorecard.reset_index()
    assert list(sc.columns) == ['feature', 'attribute', 'points']


def test_scoring_map_exists_after_fit(fitted_scoring):
    cs, data = fitted_scoring

    assert cs.scoring_map is not None
    assert isinstance(cs.scoring_map, dict)
    assert len(cs.scoring_map) == len(data.columns)
    for feature in data.columns:
        assert feature in cs.scoring_map
        for attr in data[feature].unique():
            assert attr in cs.scoring_map[feature]
