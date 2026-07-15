import pandas as pd
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score
from woe_credit_scoring.autocreditscoring import AutoCreditScoring


@pytest.fixture(scope='module')
def data():
    train = pd.read_csv('example_data/train.csv')
    valid = pd.read_csv('example_data/valid.csv')
    return train, valid


def test_autocreditscoring_pipeline(data):
    np.random.seed(42)

    train, valid = data
    varc = [v for v in train.columns if v.startswith('C_')]
    vard = [v for v in train.columns if v.startswith('D_')]

    full_train_for_acs = pd.concat([train, valid]).reset_index(drop=True)

    acs = AutoCreditScoring(
        data=full_train_for_acs,
        target='TARGET',
        continuous_features=varc,
        discrete_features=vard,
        random_state=42,
    )

    fit_params = {
        'iv_feature_threshold': 0.05,
        'max_discretization_bins': 5,
        'discretization_method': 'quantile',
        'create_reporting': False,
        'target_proportion_tolerance': 0.1,
    }
    acs.fit(**fit_params)

    
    assert acs.credit_scoring.scorecard is not None
    assert not acs.credit_scoring.scorecard.empty
    
    # Use the original validation set for prediction
    predictions = acs.predict(valid)
    assert 'score' in predictions.columns
    
    # To get probabilities, we must manually apply the pipeline and use the fitted model
    # Accessing private method for testing purposes
    valid_woe = acs._AutoCreditScoring__apply_pipeline(valid)
    valid_proba = acs.model.predict_proba(valid_woe)[:, 1]

    auc = roc_auc_score(y_true=valid['TARGET'], y_score=valid_proba)
    
    assert auc > 0.55