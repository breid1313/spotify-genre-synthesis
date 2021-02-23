# -*- coding: utf-8 -*-

import os
import pandas as pd 
from pathlib import Path

# set the location on the main data file
DATA_FILE = Path(__file__).parent.parent / "./data/data.csv"

# read the data into a DataFrame 
tracks = pd.read_csv(DATA_FILE)
print(df.head())
