
import pandas as pd
import numpy as np
import pytest
from CreditScoringToolkit import WoeEncoder

@pytest.fixture
def woe_data():
    
    data = {
        'feature': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'],
        'target':  [0, 0, 1, 0, 1, 0, 1, 1, 1]
    }
    return pd.DataFrame(data)

def test_woe_calculation(woe_data):
    
    encoder = WoeEncoder()
    encoder.fit(woe_data[['feature']], woe_data['target'])
    woe_table = pd.DataFrame.from_dict(encoder._woe_encoding_map['feature'], orient='index', columns=['woe'])
    
    
    
    # Correct WoE is log( P(0) / P(1) )
    p0 = woe_data['target'].value_counts(normalize=True)[0]
    p1 = 1 - p0
    
    # For category A
    p0_A = woe_data[woe_data['feature'] == 'A']['target'].value_counts(normalize=True)[0]
    p1_A = 1 - p0_A
    expected_A = np.log((p0_A / p0) / (p1_A / p1)) if p1_A > 0 and p0_A > 0 else 0

    # For category B
    p0_B = woe_data[woe_data['feature'] == 'B']['target'].value_counts(normalize=True)[0]
    p1_B = 1 - p0_B
    expected_B = np.log((p0_B / p0) / (p1_B / p1)) if p1_B > 0 and p0_B > 0 else 0
    
    # For category C
    p0_C = woe_data[woe_data['feature'] == 'C']['target'].value_counts(normalize=True)[0]
    p1_C = 1 - p0_C
    expected_C = np.log((p0_C / p0) / (p1_C / p1)) if p1_C > 0 and p0_C > 0 else 0

    assert np.isclose(woe_table.loc['A', 'woe'], np.log( (2/4) / (1/5)))
    assert np.isclose(woe_table.loc['B', 'woe'], np.log( (1/4) / (1/5)))
    assert np.isclose(woe_table.loc['C', 'woe'], np.log( (1/4) / (3/5)))

def test_woe_transform(woe_data):
    
    encoder = WoeEncoder()
    encoder.fit(woe_data[['feature']], woe_data['target'])
    transformed = encoder.transform(woe_data[['feature']])
    
    # Correct WoE is log(% of 0s / % of 1s)
    # P(0|A) = 2/4 = 0.5, P(1|A) = 1/5 = 0.2 -> log(0.5/0.2) is not right
    # It should be log ( (count_0 / total_0) / (count_1 / total_1) )
    total_0 = 4
    total_1 = 5
    
    p0_A = (2/total_0)
    p1_A = (1/total_1)
    expected_A = np.log(p0_A / p1_A)
    
    p0_B = (1/total_0)
    p1_B = (1/total_1)
    expected_B = np.log(p0_B / p1_B)

    p0_C = (1/total_0)
    p1_C = (3/total_1)
    expected_C = np.log(p0_C / p1_C)
    
    assert np.isclose(transformed['feature'].iloc[0], expected_A)
    assert np.isclose(transformed['feature'].iloc[3], expected_B)
    assert np.isclose(transformed['feature'].iloc[5], expected_C)

def test_woe_inverse_transform(woe_data):
    
    encoder = WoeEncoder()
    encoder.fit(woe_data[['feature']], woe_data['target'])
    transformed = encoder.transform(woe_data[['feature']])
    inversed = encoder.inverse_transform(transformed)
    
    pd.testing.assert_frame_equal(inversed, woe_data[['feature']])

def test_woe_with_missing_values():
    
    data = {
        'feature': ['A', 'A', 'B', 'MISSING'],
        'target': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    encoder = WoeEncoder()
    encoder.fit(df[['feature']], df['target'])
    transformed = encoder.transform(df[['feature']])
    
    assert 'MISSING' in encoder._woe_encoding_map['feature']
    assert not transformed.isnull().values.any()
