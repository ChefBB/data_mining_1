import numpy as np
from support.imputation import impute_outliers

outliers_columns = [
    'startYear', 'runtimeMinutes', 'numVotes', 'totalCredits',
    'criticReviewsTotal',
    'numRegions', 'userReviewsTotal', 'fill_runtimeMinutes_Bruno',
    'totalNominations', 'totalMedia', 'runtimeMinutes_notitletype'
]

power_law_columns = [
    'numVotes', 'totalCredits',
    'criticReviewsTotal', 'userReviewsTotal', 'totalNominations',
]


def detect_outliers_by_percentile(data, lower_percentile=1, upper_percentile=99):
    """
    Detects outliers in a 1D array based on percentile thresholds.

    Parameters:
        data (array-like): Input feature values.
        lower_percentile (float | None): Lower percentile threshold (default 1). Assumes None if feature is Power Law-Like.
        upper_percentile (float): Upper percentile threshold (default 99).

    Returns:
        outlier_mask (np.ndarray): Boolean array, True for outliers.
        lower_thresh (float): Value at lower percentile.
        upper_thresh (float): Value at upper percentile.
    """
    data = np.asarray(data)
    lower_thresh = np.percentile(data, lower_percentile)
    upper_thresh = np.percentile(data, upper_percentile)
    outlier_mask = (data < lower_thresh) | (data > upper_thresh)
    return outlier_mask, lower_thresh, upper_thresh


def detect_outliers(train, test, columns=outliers_columns, lower_percentile=1, upper_percentile=99):
    """
    Detects outliers in train and test datasets based on percentile thresholds.

    Parameters:
        train (array-like): Training feature values.
        test (array-like): Testing feature values.
        columns (list): List of columns to check for outliers (default: outliers_columns).
        lower_percentile (float | None): Lower percentile threshold (default 1). Assumes None if feature is Power Law-Like.
        upper_percentile (float): Upper percentile threshold (default 99).

    Returns:
        train_outlier_mask (np.ndarray): Boolean array for training data, True for outliers.
        test_outlier_mask (np.ndarray): Boolean array for testing data, True for outliers.
        lower_thresh (float): Value at lower percentile.
        upper_thresh (float): Value at upper percentile.
    """
    train['outlier'] = False
    test['outlier'] = False
    for column in columns:
        train_outlier_mask, lower_thresh, upper_thresh = detect_outliers_by_percentile(
            train[column], lower_percentile=0 if power_law_columns.__contains__(column) else lower_percentile,
            upper_percentile=upper_percentile)
        test_outlier_mask = np.logical_or(test[column] < lower_thresh, test[column] > upper_thresh)
        print(column, lower_thresh, upper_thresh)
        train[column + '_imputed'] = impute_outliers(train, lower_thresh, upper_thresh, feature=column)[column]
        test[column + '_imputed'] = impute_outliers(test, lower_thresh, upper_thresh, feature=column)[column]
        train['outlier'] = np.logical_or(train['outlier'], train_outlier_mask)
        test['outlier'] = np.logical_or(test['outlier'], test_outlier_mask)
    return train, test