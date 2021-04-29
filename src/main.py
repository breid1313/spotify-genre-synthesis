# -*- coding: utf-8 -*-

import os
import pandas as pd 
from pathlib import Path

from fuzzyCMeans import FuzzyCMeans
from agglomerative import runAgglomerative

# set the location on the data files
DATA_FILE = Path(__file__).parent.parent / "./data/data.csv"

"""
Start exploring the data.
With DataFrame.head() we get a glimpse into the first few rows.
With DataFrame.info() we can check for missing data types
and see the datatypes that read_csv() cast each column to.
"""

if __name__ == "__main__":
    # read tracks
    tracks = pd.read_csv(DATA_FILE)
    print("Track data:")
    print(tracks.head())
    print(tracks.info())

    print("Running Fuzzy C Means Clustering")
    FuzzyCMeans(DATA_FILE, sampleSize = 10000)

    print("Running Agglomerative Hierarchical Clustering")
    runAgglomerative(sample_size=20000)
