
import pandas as pd
import pytest
from CreditScoringToolkit import frequency_table


def test_frequency_table_single_variable(capsys):
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'C'],
    })
    frequency_table(df, 'category')
    captured = capsys.readouterr()
    assert 'Frequency Table for category' in captured.out


def test_frequency_table_multiple_variables(capsys):
    df = pd.DataFrame({
        'col_a': ['X', 'X', 'Y'],
        'col_b': ['M', 'N', 'M'],
    })
    frequency_table(df, ['col_a', 'col_b'])
    captured = capsys.readouterr()
    assert 'Frequency Table for col_a' in captured.out
    assert 'Frequency Table for col_b' in captured.out


def test_frequency_table_raises_type_error_on_non_dataframe():
    with pytest.raises(TypeError, match="The first argument must be a pandas DataFrame."):
        frequency_table([1, 2, 3], 'some_col')
