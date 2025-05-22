import numpy as np
from support.imputation import impute_outliers

outliers_columns = [
    'runtimeMinutes', 'numVotes', 'totalCredits',
    'criticReviewsTotal',
    'numRegions', 'userReviewsTotal', 'fill_runtimeMinutes_Bruno',
    'totalNominations', 'totalMedia', 'runtimeMinutes_notitletype'
]

power_law_columns = [
    'numVotes', 'totalCredits',
    'criticReviewsTotal', 'userReviewsTotal', 'totalNominations',
]


def detect_outliers_runtime_no_title_type(train, test, lower_percentile=1, upper_percentile=99):
    """
    Detects outliers in the 'runtimeMinutes_notitletype' column of train and test datasets.

    Parameters:
        train (array-like): Training feature values.
        test (array-like): Testing feature values.
        lower_percentile (float | None): Lower percentile threshold (default 1). Assumes None if feature is Power Law-Like.
        upper_percentile (float): Upper percentile threshold (default 99).

    Returns:
        train (pd.DataFrame): Training dataset with outliers marked and imputed.
        test (pd.DataFrame): Testing dataset with outliers marked and imputed.
    """
    train['outlier_no_type'] = False
    test['outlier_no_type'] = False
    column = 'runtimeMinutes_notitletype'
    train['type'] = train['canHaveEpisodes'].astype(int) + train['is_Short'].astype(int) * 2
    test['type'] = test['canHaveEpisodes'].astype(int) + test['is_Short'].astype(int) * 2
    for t in train['type'].unique():
        train_outlier_mask, lower_thresh, upper_thresh = detect_outliers_iqr(
            train.loc[train['type'] == t, column], lower_bound=1.5, upper_bound=1.5)
        test_outlier_mask = np.logical_or(
            test.loc[test['type'] == t, column] < lower_thresh,
            test.loc[test['type'] == t, column] > upper_thresh)
        train.loc[train['type'] == t, 'outlier_no_type'] = train_outlier_mask
        test.loc[test['type'] == t, 'outlier_no_type'] = test_outlier_mask
    train = train.drop(columns=['type'])
    test = test.drop(columns=['type'])
    return train, test


def detect_outliers_runtime(train, test, lower_percentile=1, upper_percentile=99):
    """
    Detects outliers in the 'fill_runtimeMinutes_Bruno' column of train and test datasets.

    Parameters:
        train (array-like): Training feature values.
        test (array-like): Testing feature values.
        lower_percentile (float | None): Lower percentile threshold (default 1). Assumes None if feature is Power Law-Like.
        upper_percentile (float): Upper percentile threshold (default 99).

    Returns:
        train (pd.DataFrame): Training dataset with outliers marked and imputed.
        test (pd.DataFrame): Testing dataset with outliers marked and imputed.
    """
    train['outlier_w_type'] = False
    test['outlier_w_type'] = False
    column = 'fill_runtimeMinutes_Bruno'
    for t in train['titleType'].unique():
        train_outlier_mask, lower_thresh, upper_thresh = detect_outliers_iqr(train.loc[train['titleType'] == t, column])
        test_outlier_mask = np.logical_or(
            test.loc[test['titleType'] == t, column] < lower_thresh,
            test.loc[test['titleType'] == t, column] > upper_thresh)
        train.loc[train['titleType'] == t, 'outlier_w_type'] = train_outlier_mask
        test.loc[test['titleType'] == t, 'outlier_w_type'] = test_outlier_mask
    return train, test


def detect_outliers_iqr(data, lower_bound=1.5, upper_bound=1.5):
    """
    Detects outliers in a 1D array based on IQR thresholds.

    Parameters:
        data (array-like): Input feature values.
        lower_bound (float): Lower bound multiplier for IQR (default 1.5).
        upper_bound (float): Upper bound multiplier for IQR (default 1.5).

    Returns:
        outlier_mask (np.ndarray): Boolean array, True for outliers.
        lower_thresh (float): Lower threshold value.
        upper_thresh (float): Upper threshold value.
    """
    data = np.asarray(data)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_thresh = q25 - lower_bound * iqr
    upper_thresh = q75 + upper_bound * iqr
    outlier_mask = (data < lower_thresh) | (data > upper_thresh)
    return outlier_mask, lower_thresh, upper_thresh


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
        train (pd.DataFrame): Training dataset with outliers marked and imputed.
        test (pd.DataFrame): Testing dataset with outliers marked and imputed.
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