import pandas as pd
from typing import List, Union

def frequency_table(df: pd.DataFrame, variables: Union[List[str], str]) -> None:
    """
    Displays a frequency table for the specified variables in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variables (Union[List[str], str]): List of variables (column names) to generate frequency tables for.

    Returns:
        None
    """
    if isinstance(variables, str):
        variables = [variables]
    
    for variable in variables:
        if variable not in df.columns:
            print(f"Warning: {variable} not found in DataFrame columns.")
            continue
        
        frequency_df = df[variable].value_counts().to_frame().sort_index()
        frequency_df.columns = ['Abs. Freq.']
        frequency_df['Rel. Freq.'] = frequency_df['Abs. Freq.'] / frequency_df['Abs. Freq.'].sum()
        frequency_df[['Cum. Abs. Freq.', 'Cum. Rel. Freq.']] = frequency_df.cumsum()
        
        print(f'**** Frequency Table for {variable} ****\n')
        print(frequency_df)
        print("\n" * 3)