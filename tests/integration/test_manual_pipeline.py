
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from CreditScoringToolkit import (
    DiscreteNormalizer,
    WoeEncoder,
    WoeContinuousFeatureSelector,
    WoeDiscreteFeatureSelector,
    CreditScoring
)

@pytest.fixture(scope='module')
def data():
    
    train = pd.read_csv('example_data/train.csv')
    valid = pd.read_csv('example_data/valid.csv')
    return train, valid

def test_manual_pipeline_auc(data):
    
    train, valid = data
    
    vard = [v for v in train.columns if v.startswith('D_')]
    varc = [v for v in train.columns if v.startswith('C_')]

    # 1. Normalization
    dn = DiscreteNormalizer(normalization_threshold=0.05, default_category='SMALL CATEGORIES')
    dn.fit(train[vard])
    train_norm = dn.transform(train[vard])
    valid_norm = dn.transform(valid[vard])

    # 2. Feature Selection
    wcf = WoeContinuousFeatureSelector()
    wdf = WoeDiscreteFeatureSelector()

    wcf.fit(train[varc], train['TARGET'], method='quantile', iv_threshold=0.1, max_bins=5)
    wdf.fit(train_norm, train['TARGET'], iv_threshold=0.1)

    train_selected = pd.concat([wdf.transform(train_norm), wcf.transform(train[varc])], axis=1)
    valid_selected = pd.concat([wdf.transform(valid_norm), wcf.transform(valid[varc])], axis=1)
    
    features = list(train_selected.columns)
    assert len(features) > 0

    # 3. WoE Encoding
    we = WoeEncoder()
    we.fit(train_selected, train['TARGET'])
    
    train_woe = we.transform(train_selected)
    valid_woe = we.transform(valid_selected)

    # 4. Model Training
    lr = LogisticRegression()
    lr.fit(train_woe, train['TARGET'])

    # 5. Scoring and Validation
    cs = CreditScoring()
    cs.fit(train_woe, we, lr)

    valid_pred_proba = lr.predict_proba(valid_woe)[:, 1]
    auc = roc_auc_score(y_true=valid['TARGET'], y_score=valid_pred_proba)

    assert auc > 0.65
    assert cs.scorecard is not None
    assert not cs.scorecard.empty
