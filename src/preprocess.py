import os
import pandas as pd 
import numpy as np
from scipy import stats
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def preprocess(tracks):
    """
    Check for null values. If we detect any, we should consider providing
    an estimated value or discarding the row from the analysis.
    """

    nulls = tracks.isnull()
    manual_check = []
    for row in nulls.itertuples():
        if (True in row[1:]):
            print("Null value detected at the following row: {0}.\n"
            "Manual review is required.".format(row))
            manual_check.append(row)
    if not manual_check:
        print("No null values are present in the data.")
    # delete nulls to save space. we're done with it
    del nulls


    """ 
    Check for outliers. If a row has any data that is a significant outlier, we should exclude 
    that row from our analysis. A serious outlier will affect the results of the analysis
    and prohibit a useful normalization of the data
    """
    print("Running outlier detection. For each column, rows that fall outside 3 std deviation will be discarded.\n"
    "A row will be discarded if it violates this condition for any column.")
    old_length = tracks.shape[0]
    no_normalize = ["artists", "explicit", "id", "key", "mode", "name", \
        "release_date", "year"]
    for column in tracks.columns:
        if column not in no_normalize:
            tracks[((tracks[column] - tracks[column].mean()) / tracks[column].std()).abs() < 3]
            
    new_length = tracks.shape[0]
    print("Removed {0} rows containing outliers".format(old_length - new_length))

    """
    Normalize the data. For most attributes, we should work on a consisten scale of [0,1]
    so our analysis does not yield additional weight to attributes like tempo
    or loudness that have higher numerical values.
    """
    scalar = MinMaxScaler() # defaulted to [0,1]

    # some columns dont make sense to normalize
    no_normalize = ["artists", "explicit", "id", "key", "mode", "name", \
        "release_date"]
    for column in tracks.columns:
        if column not in no_normalize:
            tracks[column] = scalar.fit_transform(np.array(tracks[column]).reshape(-1,1))
            # verify that the changes were successful
            if max(tracks[column]) > 1 or min(tracks[column]) < 0:
                print("Failed to normalize {0} column!".format(column))
    print("Skipped data normalization for these columns: {0}".format(no_normalize))
    print("Data normalized successfully!")
    print(tracks.describe(include='all'))

    return tracks
