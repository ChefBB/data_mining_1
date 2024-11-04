numeric_columns = [
        'startYear', 'endYear', 'runtimeMinutes',
        'awardWins', 'numVotes', 'worstRating',
        'numVotes', 'worstRating', 'bestRating',
        'totalImages', 'totalVideos', 'totalCredits',
        'criticReviewsTotal', 'awardNominationsExcludeWins',
        'numRegions', 'userReviewsTotal', 'ratingCount'
    ]

import pandas as pd
train_df = pd.read_csv("dm1_dataset_2425_imdb/train.csv")
test_df = pd.read_csv("dm1_dataset_2425_imdb/test.csv")

complete_df = pd.concat([train_df, test_df], ignore_index = True)

complete_df.to_csv("complete_df.csv", index=False)
