import numpy as np

# Code taken and modified from LightFM's repository (https://github.com/lyst/lightfm)
def recall_at_k(
    model,
    test_interactions,
    train_interactions=None,
    k=10,
    num_threads=1,
    check_intersections=True,
):
    """
    Measure the recall at k metric for a model: the number of positive items in
    the first k positions of the ranked list of results divided by the minimum
    of the number of positive items in the test period and k.  A perfect score
    is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the fitted model to be evaluated
    test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives in the evaluation set.
    train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
         Non-zero entries representing known positives in the train set. These
         will be omitted from the score calculations to avoid re-recommending
         known positives.
    k: integer, optional
         The k parameter.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.
    check_intersections: bool, optional, True by default,
        Only relevant when train_interactions are supplied.
        A flag that signals whether the test and train matrices should be checked
        for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

    Returns
    -------

    np.array of shape [n_users with interactions or n_users,]
         Numpy array containing recall@k scores for each user. If there are no
         interactions for a given user having items in the test period, the
         returned recall will be 0.
    """

    if num_threads < 1:
        raise ValueError("Number of threads must be 1 or larger.")

    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )

    ranks.data = np.less(ranks.data, k)

    # Assumes test_interactions only include information on positive examples
    retrieved = np.squeeze(test_interactions.getnnz(axis=1))
    users_with_test = np.flatnonzero(retrieved)
    retrieved = retrieved[users_with_test]
    hit = np.squeeze(np.array(ranks.sum(axis=1)))[users_with_test]
    weights = retrieved / retrieved.sum()
    return np.sum(weights * hit / np.minimum(retrieved, k))
