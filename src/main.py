# -*- coding: utf-8 -*-

import os
import pandas as pd 
from pathlib import Path

# set the location on the data files
DATA_FILE = Path(__file__).parent.parent / "./data/data.csv"
DATA_BY_GENRE = Path(__file__).parent.parent / "./data/data_by_genres.csv"
DATA_BY_ARTIST = Path(__file__).parent.parent / "./data/data_by_artist.csv"
DATA_BY_YEAR = Path(__file__).parent.parent / "./data/data_by_year.csv"
DATA_WITH_GENRE = Path(__file__).parent.parent / "./data/data_w_genres.csv"

"""
Start exploring the data.
With DataFrame.head() we get a glimpse into the first few rows.
With DataFrame.info() we can check for missing data types
and see the datatypes that read_csv() cast each column to.
"""

# read tracks
tracks = pd.read_csv(DATA_FILE)
print("Track data:")
print(tracks.head())
print(tracks.info())

# read tracks by artist
tracks_by_artist = pd.read_csv(DATA_BY_ARTIST)
print("\n\n\nTracks by artist")
print(tracks_by_artist.head())
print(tracks_by_artist.info())

# read tracks by genre
tracks_by_genre = pd.read_csv(DATA_BY_GENRE)
print("\n\n\nTracks by genre")
print(tracks_by_artist.head())
print(tracks_by_artist.info())

# read tracks by year
tracks_by_year = pd.read_csv(DATA_BY_YEAR)
print("\n\n\nTracks by year")
print(tracks_by_artist.head())
print(tracks_by_artist.info())

# read tracks with genre
tracks_with_genre = pd.read_csv(DATA_WITH_GENRE)
print("\n\n\nTracks with genre")
print(tracks_by_artist.head())
print(tracks_by_artist.info())
