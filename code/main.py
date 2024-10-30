import pandas as pd
import numpy as np

from constants import numeric_columns as num_cols


train_df = pd.read_csv('dm1_dataset_2425_imdb/train.csv')
test_df = pd.read_csv('dm1_dataset_2425_imdb/test.csv')

# train_df[num_cols] = train_df[num_cols].apply(
    # pd.to_numeric, errors='coerce'
    # )

for col in num_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
# train_df[num_cols] = pd.to_numeric(train_df[num_cols], errors='coerce')

train_df.loc[train_df['genres'] == r'\N', 'genres'] = np.nan

# train_df