
import pandas as pd
import numpy as np
import pytest
from CreditScoringToolkit import WoeDiscreteFeatureSelector, WoeContinuousFeatureSelector

@pytest.fixture
def feature_selection_data():
    
    data = {
        'discrete_feature_good': ['A'] * 5 + ['B'] * 5,
        'discrete_feature_bad':  ['C'] * 9 + ['D'] * 1,
        'continuous_feature_good': np.arange(10),
        'continuous_feature_bad':  np.concatenate([np.zeros(9), np.ones(1)]),
        'target': [0,0,0,0,1, 0,1,1,1,1] 
    }
    return pd.DataFrame(data)

def test_woe_discrete_feature_selector(feature_selection_data):
    
    selector = WoeDiscreteFeatureSelector()
    selector.fit(
        feature_selection_data[['discrete_feature_good', 'discrete_feature_bad']],
        feature_selection_data['target'],
        iv_threshold=0.1
    )
    
    assert 'discrete_feature_good' in selector.selected_features
    assert 'discrete_feature_bad' not in selector.selected_features
    
    transformed = selector.transform(feature_selection_data[['discrete_feature_good', 'discrete_feature_bad']])
    assert list(transformed.columns) == ['discrete_feature_good']

def test_woe_continuous_feature_selector(feature_selection_data):
    
    selector = WoeContinuousFeatureSelector()
    selector.fit(
        feature_selection_data[['continuous_feature_good', 'continuous_feature_bad']],
        feature_selection_data['target'],
        iv_threshold=0.1, 
        method='quantile', 
        max_bins=2
    )
    
    assert len(selector.selected_features) == 1
    assert selector.selected_features[0]['root_feature'] == 'continuous_feature_good'
    
    transformed = selector.transform(feature_selection_data[['continuous_feature_good', 'continuous_feature_bad']])
    
    
    assert any(col.startswith('disc_continuous_feature_good') for col in transformed.columns)
    assert not any(col.startswith('disc_continuous_feature_bad') for col in transformed.columns)
