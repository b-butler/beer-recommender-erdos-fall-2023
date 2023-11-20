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

    users = [0]*int_matrix.shape[0]
    test_sets = [[] for _ in range(n_splits)]
    train_set = []
    # for each user, separate their reviews into a train set and different test sets.
    for i in range(len(users)):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        users[i] = indices[row_start:row_end] # these are the beers that user i reviewed
        # place the users in buckets based on number of reviews
        if len(users[i]) < 5:
            for beer_idx in users[i]:
                train_set.append([i,beer_idx])
        elif len(users[i]) < 11:
            users[i] = np.random.permutation(users[i])
            for j in range(n_splits):
                test_sets[j].append([i,users[i][j]])
            for beer_idx in users[i][n_splits:]:
                train_set.append([i,beer_idx])
        else :
            n = len(users[i])
            k = n//10
            users[i] = np.random.permutation(users[i])
            for j in range(n_splits):
                for beer_idx in users[i][j*k:(j+1)*k]:
                    test_sets[j].append([i, beer_idx])
            for beer_idx in users[i][n_splits*k:]:
                train_set.append([i, beer_idx])
    #train_set = np.array(train_set)
    #test_sets = [np.array(test_sets[i]) for i in range(len(test_sets))]

    splits = [0]*n_splits
    for j in range(n_splits):
        train_j = train_set.copy()
        for i in range(n_splits):
            if i != j:
                train_j.extend(test_sets[i])
        for coords in train_j:
            if len(coords) != 2:
                print("UH OH",coords)
        splits[j] = (np.array(train_j), np.array(test_sets[j]))
    return splits