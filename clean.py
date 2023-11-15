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


def remove_duplicate_reviews(data, columns=("review_profilename", "beer_beerid")):
    """Remove duplicate reviews defined by columns.

    Parameters
    ----------
    data: pandas.DataFrame
        The Beer Advocate data frame.
    columns: Sequence[str], optional
        The columns to compare for duplicates.
    """
    return data.drop_duplicates(columns)


# EXACT duplicate beers are beers that agree on beer_beerid, beer_name, beer_style, beer_abv, brewery_name, and brewery_id
# INEXACT duplicate beers are beers that agree on beer_name and brewery_name (but differ on some other data).
def get_beers(data):
    """From the reviews data, just get a dataframe of the beers.

    Parameters
    ----------
    data: pandas.DataFrame
        The Beer Advocate data frame.
    """
    beers = data.copy().drop(['review_time',
                              'review_overall',
                              'review_aroma',
                              'review_appearance',
                              'review_profilename',
                              'review_palate',
                              'review_taste'], axis = 1) # drop all columns having to do with a specific review
    beers = beers.drop_duplicates() # and then eliminate EXACT duplicates just keeping the first instance of each
    return beers
    
def remove_dup_beer_rows(data, dup_params=("beer_name", "brewery_name")):
    """Remove rows where the review is of an inexact duplicate beer.
    By inexact, we mean they agree on only some subset of the parameters.
    By default we will mean they agree on beer_name and brewery_name.

    Parameters
    ----------
    data: pandas.DataFrame
        The Beer Advocate data frame.
    columns: Sequence[str], optional
        The columns where agreement determines duplicates. Defaults to
        "beer_name" and "brewery_name".
    """
    beers = get_beers(data) # get all the beers
    
    # Then extract all beer_beerids which occur as INEXACT duplicates.
    inexact_dup_ids = beers[beers.duplicated(dup_params, keep = False)]['beer_beerid']
    
    return data[~data.beer_beerid.isin(inexact_dup_ids)]

def merge_similar_name_breweries(data):
    """Clean the reviews data by merging similar named breweries to have one common name.
    The list of breweries for which we do this is not expected to be exhaustive.
    
    Parameters
    ----------
    data: pandas.DataFrame
        The Beer Advocate data frame.
    """
    similar_names = [["BJ's Restaurant & Brewhouse",
                      "BJ's Restaurant And Brewhouse",
                      "BJ's Restaurant & Brewery"],
                    ["Hops Grillhouse & Brewery",
                     "Hops Grill & Brewery"],
                    ["Rock Bottom Restaurant & Brewery",
                     "Rock Bottom Restaurant and Brewery"],
                    ["Les Trois Brasseurs",
                     "Les 3 Brasseurs"]]
    for similar_name in similar_names:
            data.loc[data.brewery_name.isin(similar_name[1:]),'brewery_name'] = similar_name[0]
    return data
    
def merge_brewery_ids(data):
    """ Assign a single brewery id to breweries that have the same name.
    
    Parameters
    ----------
    data: pandas.DataFrame
        The Beer Advocate data frame.
    """
    breweries = data[['brewery_name','brewery_id']].drop_duplicates()
    duplicates = breweries.brewery_name.value_counts().loc[breweries.brewery_name.value_counts()>1]
    # go through the reviews and update the brewery_id to be the largest id corresponding to a given name
    for duplicate in duplicates.index:
        data.loc[data.brewery_name==duplicate,'brewery_id'] = breweries.loc[breweries.brewery_name==duplicate,
                                                                             'brewery_id'].max()
    return data