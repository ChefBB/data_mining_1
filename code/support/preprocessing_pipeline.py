"""
This module contains functions that assist with the preprocessing of training and testing datasets.

The functions in this file are designed to streamline and standardize the preprocessing steps
required for machine learning pipelines, ensuring consistency and reproducibility.


Steps:
1. Initialization of the datasets from starting csvs.
2. Applies imputation to the datasets.
3. Applies transformations to the datasets.
"""

import pandas as pd
from df_init import init
# from outlier_detection.outlier_master import outlier_detection
from imputation import impute_data
from transformations import apply_transformations
from embedding import embedding


def preprocess_train_test(train: str, test: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    !!! NOT FULLY IMPLEMENTED YET !!!
    
    Initializes and reprocesses the training and testing datasets.

    ----------
    Parameters
    ----------
    train : str
        The path to the training dataset.
        
    test : str
        The path to the testing dataset.
        
    ----------
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]: The preprocessed training and testing datasets.
    """
    
    # Load the datasets
    train_df = init(train)
    test_df = init(test)
    
    # Perform preprocessing steps
    # TODO: Implement outlier detection pipeline
    # train_df = outlier_detection(train_df, test_df)
    
    train_df, test_df = impute_data(train_df, test_df)
    
    train_df, test_df = embedding(train_df, test_df)
    
    train_df, test_df = apply_transformations(train_df, test_df)
    
    return train_df, test_df