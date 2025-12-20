
import pandas as pd
import numpy as np
import pytest
from CreditScoringToolkit import DiscreteNormalizer

@pytest.fixture
def sample_data():
    
    data = {
        'feature1': ['A', 'A', 'B', 'B', 'B', 'C', 'D', 'D', np.nan],
        'feature2': ['X', 'X', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z']
    }
    return pd.DataFrame(data)

def test_small_category_aggregation(sample_data):
    
    dn = DiscreteNormalizer(normalization_threshold=0.3, default_category='SMALL')
    dn.fit(sample_data[['feature1']])
    transformed = dn.transform(sample_data[['feature1']])
    expected_values = ['SMALL', 'SMALL', 'B', 'B', 'B', 'SMALL', 'SMALL', 'SMALL', 'SMALL']
    assert transformed['feature1'].tolist() == expected_values

def test_missing_value_handling(sample_data):
    
    dn = DiscreteNormalizer(normalization_threshold=0.1)
    dn.fit(sample_data[['feature1']])
    transformed = dn.transform(sample_data[['feature1']])
    assert 'MISSING' in transformed['feature1'].unique()
    assert transformed['feature1'].iloc[8] == 'MISSING'

def test_unseen_categories():
    
    train_data = pd.DataFrame({'feature1': ['A', 'A', 'B', 'B', 'B']})
    test_data = pd.DataFrame({'feature1': ['A', 'C', 'B']})
    dn = DiscreteNormalizer()
    dn.fit(train_data)
    transformed = dn.transform(test_data)
    
    assert transformed['feature1'].tolist() == ['A', 'B', 'B']

def test_no_small_categories(sample_data):
    
    dn = DiscreteNormalizer(normalization_threshold=0.1)
    dn.fit(sample_data[['feature2']])
    transformed = dn.transform(sample_data[['feature2']])
    assert 'SMALL CATEGORIES' not in transformed['feature2'].unique()
    expected_values = ['X', 'X', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z']
    assert transformed['feature2'].tolist() == expected_values
