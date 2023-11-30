"""Methods to execute our validation scheme"""
import numpy as np


def get_splits(int_matrix, random_state=42):
    """Returns the matrix coordinates for 5 train and test splits.

    Parameters
    ----------
    int_matrix: 2d matrix
    random_state: int, optional
    """
    n_splits = 5
    np.random.seed(random_state)
    # Convert the dense matrix to a CSR sparse matrix
    int_matrix = int_matrix.tocsr()

    # Accessing the CSR components
    data = int_matrix.data
    indices = int_matrix.indices
    indptr = int_matrix.indptr

    N_users = int_matrix.shape[0]
    test_sets = [[] for _ in range(n_splits)]
    train_set = []  # this set is never in the test set
    # for each user, separate their reviews into a train set and different test sets.
    for i in range(N_users):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        user_reviews = indices[
            row_start:row_end
        ]  # these are the beers that user i reviewed
        n_reviews = len(user_reviews)
        # place the users in buckets based on number of reviews
        if n_reviews < 5:
            for beer_idx in user_reviews:
                train_set.append([i, beer_idx])
        elif n_reviews < 11:
            user_reviews = np.random.permutation(user_reviews)
            for j in range(n_splits):
                test_sets[j].append([i, user_reviews[j]])
            for beer_idx in user_reviews[n_splits:]:
                train_set.append([i, beer_idx])
        else:
            k = n_reviews // 10
            user_reviews = np.random.permutation(user_reviews)
            for j in range(n_splits):
                for beer_idx in user_reviews[j * k : (j + 1) * k]:
                    test_sets[j].append([i, beer_idx])
            for beer_idx in user_reviews[n_splits * k :]:
                train_set.append([i, beer_idx])
    # create the splits by appending all but one test set to the train set
    splits = []
    for j in range(n_splits):
        train_j = train_set.copy()
        for i in range(n_splits):
            if i != j:
                train_j.extend(test_sets[i])
        splits.append((np.array(train_j), np.array(test_sets[j])))
    return splits
