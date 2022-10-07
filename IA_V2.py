# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:50:07 2022

@author: disrct
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from time import process_time
import pandas as pd
import numpy as np


def applyArtistPopularity(artist):
    try: return artistPopularity[artist]
    except: return np.nan

def applyArtistFollowers(artist):
    try: return artistFollowers[artist]
    except: return np.nan

inicio = process_time()
tracksCols = [
        "popularity", "duration_ms", "explicit", "artists", "danceability",
        "energy", "key", "loudness", "mode", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "time_signature"
        ]

artistsCols = ["followers", "genres", "name", "popularity"]


tracks = pd.read_csv("data/tracks.csv", usecols=tracksCols)
artists = pd.read_csv("data/artists.csv")


tracks["artists"] = tracks["artists"].str.strip("[]")
tracks["artists"] = tracks["artists"].str.replace("'", "", regex=True)
tracks["artists"] = tracks["artists"].str.replace("\"", "", regex=True)
tracks["artists"] = tracks["artists"].str.split(", ")
tracks = tracks.explode("artists").reset_index()


musicPopularity = tracks.groupby("artists")["popularity"].mean()
artistPopularity = artists.groupby("name")["popularity"].max()
artistFollowers = artists.groupby("name")["followers"].max()



tracks["musicPopularity"] = tracks["artists"].apply(lambda artist: musicPopularity[artist])
tracks["artistPopularity"] = tracks["artists"].apply(applyArtistPopularity)
tracks["artistFollowers"] = tracks["artists"].apply(applyArtistFollowers)
tracks = tracks.dropna()


xCols = [
        'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'time_signature', 'musicPopularity',
        'artistPopularity', 'artistFollowers'
        ]

x = tracks[xCols].values
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

y = tracks["popularity"].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
y_train = np.ravel(y_train)

rfr = RandomForestRegressor().fit(x_train, y_train)
print("Acurácia RFR:", round(rfr.score(x_test, y_test), 4) * 100)

print(process_time() - inicio)