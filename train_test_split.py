"""Methods to execute our validation scheme"""

import pandas as pd
import numpy as np



def sample_1(group, random_state):
    return group.sample(1, random_state = random_state)
def sample_10_percent(group, random_state):
    n = group.v_counts.iloc[0]
    return group.sample(n//10, random_state = random_state)

def get_splits(cleaned_df, random_state = 42):
    """ Takes in a cleaned dataframe. Creates the reviewer_id and beer_id in the same manner as is done for the interaction matrix and adds these as columns.
    
    Returns 5 disjoint train and test sets.
    """
    n_splits = 5
    cleaned_df['reviewer_id'] = cleaned_df["review_profilename"].astype('category').cat.codes
    cleaned_df['beer_id'] = cleaned_df["beer_beerid"].astype("category").cat.codes

    cleaned_df['v_counts'] = cleaned_df.reviewer_id.map(cleaned_df.reviewer_id.value_counts())
    
    low_reviewers = cleaned_df.loc[cleaned_df.v_counts < 5]
    mid_reviewers = cleaned_df.loc[(cleaned_df.v_counts >=5) & (cleaned_df.v_counts <= 10)]
    high_reviewers = cleaned_df.loc[cleaned_df.v_counts > 10]
    
    test_sets = [np.empty((0,2), dtype=int)]*n_splits
    for i in range(n_splits):
        test_i = mid_reviewers.groupby('reviewer_id', group_keys=False).apply(sample_1,
                                                                              random_state=random_state)
        mid_reviewers = mid_reviewers.drop(test_i.index)
        test_sets[i] = np.append(test_sets[i],test_i[['reviewer_id','beer_id']].values, axis=0)

        test_i = high_reviewers.groupby('reviewer_id', group_keys=False).apply(sample_10_percent,
                                                                               random_state=random_state)
        high_reviewers = high_reviewers.drop(test_i.index)
        test_sets[i] = np.append(test_sets[i],test_i[['reviewer_id','beer_id']].values, axis=0)
    
    train = np.append(low_reviewers[['reviewer_id','beer_id']].values, 
                      mid_reviewers[['reviewer_id','beer_id']].values,
                      axis=0)
    train = np.append(train,
                      high_reviewers[['reviewer_id','beer_id']].values,
                      axis=0)
    
    splits = [0]*5
    for i in range(n_splits):
        train_i = train.copy()
        for j in range(n_splits):
            if i != j:
                train_i = np.append(train_i, test_sets[j], axis=0)
        splits[i] = (train_i, test_sets[i])
    return splits