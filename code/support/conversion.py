"""
This module contains functions to convert data from one format to another
"""

import ast

feats_to_list = [
    'countryOfOrigin',
]


def convert_string_list(str_list: str) -> list[str]:
    """
    Converts a string in the format "['<str>' (, ...)]" to a list

    Args:
        str_list (str): string in the format "['<str>' (, ...)]"

    Returns:
        list[str]: the formatted data
    """
    stripped = str_list.strip()
    if stripped == '[]':
        return []
    
    country_list = ast.literal_eval(stripped)  # Convert string to actual list
    return [f'{item}' for item in country_list]