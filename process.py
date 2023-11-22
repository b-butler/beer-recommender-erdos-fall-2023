"""Classes and functions to process the data into ML ready forms."""

import numpy as np
import scipy as sp


class InteractionMatrixTransformer:
    """Transform Beer Advocate dataset to interaction matrix.

    Performs some preliminary cleaning/modification.

    - Remove null reviewer profile names
    - Transform beer_beerid to contiguous values from [0, num_of_beers).
    """
    def __init__(self, data):
        self._data = data
        self._add_matrix_indices(self._data)
        self._shape = (
            self._reviewer_ids.max() + 1, self._beer_ids.max() + 1)

    def _add_matrix_indices(self, data):
        self._reviewer_ids = data["review_profilename"].astype('category').cat.codes
        # Necessary to make ids contiguous from zero
        self._beer_ids = data["beer_beerid"].astype("category").cat.codes

    def fit(self, method="positive_negative", threshold=2.5, review="overall"):
        if method == "positive_negative":
            return self.to_positive_negative(threshold, review)
        if method == "zero_one":
            return self.to_zero_one(threshold, review)
        if method == "range":
            return self.to_range(review)
        raise ValueError(f"Unknown method ({method}).")

    @property
    def _indices(self):
        return (self._reviewer_ids, self._beer_ids)

    def to_zero_one(self, threshold=2.5, review="overall"):
        """Transform to 0, 1 matrix with given threshold.

        Parameters
        ----------
        threshold: int, optional
            The cut-off for review values to turn to one. Defaults to 2.5.
        review: string, optional
            The review to use. Defaults to "overall".
        """
        interactions = self._data["review_" + review].values
        interactions = np.where(
            interactions >= threshold, 1, 0)
        return sp.sparse.csr_matrix((interactions, self._indices), self._shape, copy=True)


    def to_positive_negative(self, threshold=2.5, review="overall"):
        """Transform to -1, 0, 1 matrix with given threshold.

        Parameters
        ----------
        threshold: int, optional
            The cut-off for review values to turn to one and
            which to turn to -1. Defaults to 2.5.
        review: string, optional
            The review to use. Defaults to "overall".
        """
        interactions = self._data["review_" + review].values
        interactions = np.where(
            interactions >= threshold, 1, -1)
        return sp.sparse.csr_matrix((interactions, self._indices), self._shape, copy=True)


    def to_range(self, review="overall"):
        """Transform to continuous matrix.

        Parameters
        ----------
        review: string, optional
            The review to use. Defaults to "overall".
        """
        interactions = self._data["review_" + review].values
        return sp.sparse.csr_matrix((interactions, self._indices), self._shape)
