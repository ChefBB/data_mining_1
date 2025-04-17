"""
This module contains functions for handling the
'countryOfOrigin' and 'regions' columns.
"""

import pycountry_convert as pc
import pandas as pd
from collections import Counter


# List of continents
CONTINENTS = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA', 'UNK']


def init_countries_trn_tst(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Initializes the 'countryOfOrigin' and 'regions' columns for the training and testing datasets.
    This includes converting country names to ISO 3166-1 alpha-2 codes and handling missing values.
    
    Parameters
    ----------
    train : pd.DataFrame
        The training dataset.
        
    test : pd.DataFrame
        The testing dataset.
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]: The modified training and testing datasets.
    """
    train_new_feats, freq_enc = explode_continents_and_freq_enc(train)
    test_new_feats, _ = explode_continents_and_freq_enc(test, freq_enc)
    train = pd.concat([train, train_new_feats], axis=1)
    test = pd.concat([test, test_new_feats], axis=1)
    
    return train, test


def country_to_continent(country_code):
    """
    Converts the country code to the continent name.
    """
    try:
        return pc.country_alpha2_to_continent_code(country_code)
    except KeyError:
        return 'UNK'
    
    
def get_encoded(country_list: list[str], name: str = '') -> dict[str, int]:
    """
    Encodes the series of countries to continents and returns a dictionary
    with the number of occurrences for each continent.
    
    ----------
    Parameters
    ----------
    country_list : list[str]
        The list of countries
        
    name : str | None
        The name to prepend to the continent name
        
    Returns
    -------
    dict[str, int]: The dictionary with the number of occurrences for each continent
    """
    # Map each country to its continent
    continents = [country_to_continent(country) for country in country_list]
    
    # Increment occurrences of each continent
    continent_counts = Counter(continents)
    
    continent_counts = {f"{name}_{key}": value for key, value in continent_counts.items()}
    for continent in CONTINENTS:
        if (name + '_' + continent) not in continent_counts:
            continent_counts[f"{name}_{continent}"] = 0
    
    return dict(continent_counts)


def explode_continents_and_freq_enc(df: pd.DataFrame, freq_enc: dict[str, float] | None=None) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Explodes the 'countryOfOrigin', 'regions' columns into separate rows for each continent.
    Counts the number of occurrences for each continent representative.
    Also frequency-encodes the 2 columns.
    
    For test set usage, use frequency encoding obtained on train set.
    
    Returns the modified dataframe.
    
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
        
    freq_enc : dict[str, float] | None
        The frequency encoding for the columns
        If None, it will be calculated.
        If not None, it will be used to map the values.
        
    Returns
    -------
    pd.DataFrame: The new columns
    
    dict[str, float]: The frequency encoding for the columns
    ----------
    """
    result = pd.DataFrame()
    flag = False
    if freq_enc is None:
        freq_enc = {}
        flag = True
    # Explode the column to handle lists
    exploded = df['countryOfOrigin'].explode()
    
    # Get frequency encoding for the column
    if flag:
        freq_enc['countryOfOrigin'] = exploded.value_counts(normalize=True).to_dict()
    result['countryOfOrigin_freq_enc'] = df['countryOfOrigin'].map(
        lambda x, freq_enc=freq_enc['countryOfOrigin']: sum(freq_enc.get(i, 0) for i in x))
    
    # Apply the get_encoded function to the column
    encoded = df['countryOfOrigin'].apply(get_encoded, name='countryOfOrigin')
    
    # Append to result dataframe the new columns
    for key in encoded[0].keys():
        result[key] = [enc[key] for enc in encoded]
    
    return result, freq_enc