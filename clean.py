"""Functions for cleaning the Beer Advocate data before processing."""
def remove_null_rows(data, columns=("review_profilename", "review_overall")):
    """Remove rows with a null in given columns.

    Parameters
    ----------
    data: pandas.DataFrame
        The Beer Advocate data frame.
    columns: Sequence[str], optional
        The columns where a null invalidates the row. Defaults to
        "review_profilename" and "review_overall".
    """
    return data.dropna(axis="index", subset=columns)

