"""
This file contains various preprocessing transformations for
the dataset
"""
import pandas as pd
import numpy as np


to_log = [
    'criticReviewsTotal', 'userReviewsTotal',
    'numRegions',
    'totalMedia',
    'totalNominations', 'totalCredits',
    # 'regions_freq_enc', 'regions_EU', 'regions_NA', 'regions_AS',
    # 'regions_AF', 'regions_OC', 'regions_SA', 'regions_UNK',
    # 'countryOfOrigin_freq_enc', 'countryOfOrigin_NA', 'countryOfOrigin_AF',
    # 'countryOfOrigin_AS', 'countryOfOrigin_EU', 'countryOfOrigin_OC',
    # 'countryOfOrigin_SA', 'countryOfOrigin_UNK'
]

def apply_transformations(
    train: pd.DataFrame, test: pd.DataFrame | None=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Applies transformations to the training and testing datasets.
    This includes log transformations on specified columns.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame | None, optional): The testing dataset. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: The transformed training and testing datasets.
    """
    return apply_log_scale(train, to_log), apply_log_scale(test, to_log) if test is not None else None


def apply_log_scale(df: pd.DataFrame, columns: list[str]=to_log) -> pd.DataFrame:
    """
    Applies log transformation to the specified columns in the dataframe.
    
    Parameters:
    ----------
    df (pd.DataFrame): Input dataframe.
    columns (list): List of column names to apply log transformation.
    
    
    
    Returns:
    ----------
    pd.DataFrame: Dataframe with log-transformed columns.
    """
    result = df.copy()
    for col in columns:
        if col in df.columns:
            result[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    return result