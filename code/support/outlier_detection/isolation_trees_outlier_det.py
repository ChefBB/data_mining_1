"""
Some constants and functions to help with outliers detection
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from h2o.estimators.extended_isolation_forest import H2OExtendedIsolationForestEstimator
import h2o
from h2o.frame import H2OFrame




feats_to_keep_iso_forest = [
    'runtimeMinutes', 'totalCredits', 'reviewsTotal',
    'numRegions', 'ratingCount', 'castNumber',
    'companiesNumber', 'averageRating', 'writerCredits',
    'directorsCredits', 'totalMedia', 'totalNominations',
    'regions_freq_enc',
    'regions_EU', 'regions_NA', 'regions_AS', 'regions_AF', 'regions_OC',
    'regions_SA', 'regions_UNK',
    'countryOfOrigin_freq_enc',
    'countryOfOrigin_NA', 'countryOfOrigin_AF', 'countryOfOrigin_AS',
    'countryOfOrigin_EU', 'countryOfOrigin_OC', 'countryOfOrigin_SA',
    'countryOfOrigin_UNK'
]


def classwise_iso_forest(df: pd.DataFrame,
                         feats: list[str]=feats_to_keep_iso_forest,
                         ) -> tuple[pd.Series, dict]:
    """
    Applies Isolation Forest to detect outliers in the DataFrame.
    The function scales the data and applies Isolation Forest separately for each titleType.
    The outlier predictions are stored in a new column 'outlier'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    feats : list[str], optional
        The features to use for the model. Defaults to feats_to_keep_iso_forest.
    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.

    dict: A dictionary containing the Isolation Forest models for each titleType.
    """
    # Initialize a list to store processed groups
    processed_groups = []
    # Initialize Isolation Forest dictionary
    iso_forest_dict = {}

    # Group by titleType and apply Isolation Forest
    for _, group in df.groupby('titleType'):
        # Preserve the original index
        original_index = group.index
        group = group.reset_index(drop=True)

        # Apply Isolation Forest
        iso_forest = (
            IsolationForest(random_state=42, contamination=0.01))
        group['outlier'] = iso_forest.fit_predict(group[feats])

        iso_forest_dict[group['titleType'].iloc[0]] = iso_forest
        
        # Restore the original index
        group.index = original_index
        processed_groups.append(group)

    # Concatenate all groups and sort by the original index
    result = pd.concat(processed_groups).sort_index()

    return result['outlier'], iso_forest_dict


def global_iso_forest(df: pd.DataFrame,
                            feats: list[str]=feats_to_keep_iso_forest)->tuple[pd.Series, dict]:
    """
    Applies Isolation Forest to detect outliers in the entire DataFrame.
    The outlier predictions are stored in a new column 'outlier'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.

    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.
    
    dict: A dictionary containing the Isolation Forest model for the entire dataset.
    """
    # Preserve the original index
    original_index = df.index

    # Reset index for processing
    df = df.reset_index(drop=True)

    # Apply Isolation Forest
    iso_forest = IsolationForest(random_state=42, contamination=0.01)
    df['outlier'] = iso_forest.fit_predict(df[feats])

    # Restore the original index
    df.index = original_index

    return df['outlier'], iso_forest


def global_extended_iso_forest(df: pd.DataFrame, feats: list[str] = feats_to_keep_iso_forest) -> tuple[pd.Series, H2OExtendedIsolationForestEstimator]:
    """
    Applies H2O Extended Isolation Forest to detect outliers in the DataFrame.
    The outlier predictions are stored in a new column 'outlier_h2o'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    feats : list[str], optional
        The features to use for the model. Defaults to feats_to_keep_iso_forest.

    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.
    H2OExtendedIsolationForestEstimator: The trained H2O Extended Isolation Forest model.
    """

    # Initialize H2O
    h2o.init()

    # Convert DataFrame to H2OFrame
    h2o_df = H2OFrame(df[feats])

    # Initialize the H2O Extended Isolation Forest model
    eif_model = H2OExtendedIsolationForestEstimator(seed=42, sample_size=256, ntrees=100)

    # Train the model
    eif_model.train(training_frame=h2o_df)

    # Predict outliers
    predictions = eif_model.predict(h2o_df)

    # Extract the anomaly score
    anomaly_scores = predictions['anomaly_score'].as_data_frame().values.flatten()

    # Determine the threshold for the top 1% of outliers
    threshold = pd.Series(anomaly_scores).quantile(0.99)

    # Assign -1 to the top 1% of outliers and 1 to the rest
    result = pd.Series((anomaly_scores >= threshold).astype(int)).replace({0: 1, 1: -1})

    # Shutdown H2O
    h2o.shutdown(prompt=False)

    return result, eif_model


def classwise_extended_iso_forest(df: pd.DataFrame, feats: list[str] = feats_to_keep_iso_forest) -> tuple[pd.Series, dict]:
    """
    Applies H2O Extended Isolation Forest to detect outliers in the DataFrame.
    The outlier predictions are stored in a new column 'outlier_h2o'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    feats : list[str], optional
        The features to use for the model. Defaults to feats_to_keep_iso_forest.

    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.
    dict: A dictionary containing the H2O Extended Isolation Forest models for each titleType.
    """
    processed_groups = []
    # Initialize Isolation Forest dictionary
    iso_forest_dict = {}

    # Initialize H2O
    h2o.init()

    for _, group in df.groupby('titleType'):
        # Preserve the original index
        original_index = group.index
        group = group.reset_index(drop=True)
        
        # Convert DataFrame to H2OFrame
        h2o_df = H2OFrame(group[feats])

        # Initialize the H2O Extended Isolation Forest model
        eif_model = H2OExtendedIsolationForestEstimator(seed=42, sample_size=256, ntrees=100)

        # Train the model
        eif_model.train(training_frame=h2o_df)

        # Predict outliers
        predictions = eif_model.predict(h2o_df)

        # Extract the anomaly score
        anomaly_scores = predictions['anomaly_score'].as_data_frame().values.flatten()

        # Determine the threshold for the top 1% of outliers
        threshold = pd.Series(anomaly_scores).quantile(0.99)
        
        group['outlier'] = pd.Series((anomaly_scores >= threshold).astype(int)).replace({0: 1, 1: -1})

        # Restore the original index
        group.index = original_index
        processed_groups.append(group)
        iso_forest_dict[group['titleType'].iloc[0]] = eif_model

    # Assign -1 to the top 1% of outliers and 1 to the rest
    result = pd.concat(processed_groups).set_index(df.index)['outlier']

    # Shutdown H2O
    h2o.shutdown(prompt=False)

    return result, iso_forest_dict
     
    
def apply_iso_forest(df: pd.DataFrame, model: IsolationForest | H2OExtendedIsolationForestEstimator |
                     dict[str, IsolationForest] | dict[str, H2OExtendedIsolationForestEstimator],
                     feats: list[str]=feats_to_keep_iso_forest) -> pd.DataFrame:
    """
    !!!NOT TESTED!!!

    Applies the Isolation Forest model to the DataFrame.
    If a dictionary of models is provided, it applies the model corresponding to the titleType.
    Otherwise, it applies the global model.
    The outlier predictions are stored in a new column 'outlier'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    model : IsolationForest | dict[str, IsolationForest]
        The Isolation Forest model or a dictionary of models for each titleType.
    
    Returns
    -------
    pd.DataFrame: The DataFrame with the outlier predictions added.
    """
    result = df.copy()
    # Check if model is a dictionary
    if isinstance(model, dict[str, IsolationForest]):
        # Apply the model corresponding to the titleType
        result['outlier'] = result.apply(
            lambda row: model[row['titleType']].predict([row[feats]])[0], axis=1)
    if isinstance(model, dict[str, H2OExtendedIsolationForestEstimator]):
        # Convert DataFrame to H2OFrame
        h2o_df = H2OFrame(result[feats])

        # Apply the model corresponding to the titleType
        result['outlier'] = result.apply(
            lambda row: model[row['titleType']].predict(h2o_df[row.name, :])['anomaly_score'].as_data_frame().values[0][0], axis=1)
    elif isinstance(model, H2OExtendedIsolationForestEstimator):
        # Convert DataFrame to H2OFrame
        h2o_df = H2OFrame(result[feats])

        # Predict outliers
        predictions = model.predict(h2o_df)

        # Extract the anomaly score
        anomaly_scores = predictions['anomaly_score'].as_data_frame().values.flatten()

        # Determine the threshold for the top 1% of outliers
        threshold = pd.Series(anomaly_scores).quantile(0.99)

        # Assign -1 to the top 1% of outliers and 1 to the rest
        result['outlier'] = pd.Series((anomaly_scores >= threshold).astype(int)).replace({0: 1, 1: -1})
    else:
        # Apply the global model
        result['outlier'] = model.predict(result[feats])

    return result