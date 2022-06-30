import pandas as pd
def frequency_table(df:pd.DataFrame,var:list):
    """Displays a frequency table 

    Args:
        df (pd.DataFrame): Data
        var (list): List of variables 
    """
    if type(var)==str:
        var = [var]
    for v in var:
        aux = df[v].value_counts().to_frame().sort_index()
        aux.columns = ['Abs. Freq.']
        aux['Rel. Freq.'] = aux['Abs. Freq.']/aux['Abs. Freq.'].sum()
        aux[['Cumm. Abs. Freq.','Cumm. Rel. Freq.']] = aux.cumsum()
        print(f'****Frequency Table  {v}  ***\n\n')
        print(aux)
        print("\n"*3)
