"""
This module is used to initialize the dataframe.
"""


import support.conversion as cv
import pandas as pd

def init(path: str = 'pp_train.csv') -> pd.DataFrame:
    """
    Initializes dataframe from the input csv format.
    """
    df = pd.read_csv(path)
    
    # Handling of lists
    for feat in cv.feats_to_list:
        df[feat] = df[feat].apply(cv.convert_string_list)
        
    return df