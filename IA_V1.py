from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from time import process_time
import pandas as pd
import numpy as np

inicio = process_time()
cols = [
        "valence", "acousticness", "artists", "danceability", 
        "energy", "explicit", "instrumentalness", "liveness", 
        "loudness", "mode", "popularity", "speechiness", "tempo"
        ]

df = pd.read_csv("data/data.csv", usecols=cols)


df["artists"] = df["artists"].str.strip("[]")
df["artists"] = df["artists"].str.replace("'", "", regex=True)
df["artists"] = df["artists"].str.split(", ")
df = df.explode("artists").reset_index()

artistsPopularity = df.groupby("artists")["popularity"].mean()
df["artistsPopularity"] = df["artists"].apply(lambda row: artistsPopularity[f'{row}'])

xCols = [
        "valence", "acousticness", "artistsPopularity", "danceability", 
        "energy", "explicit", "instrumentalness", "liveness", 
        "loudness", "mode", "speechiness", "tempo"
        ]

x = df[xCols].values
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

y = df["popularity"].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
y_train = np.ravel(y_train)

rfr = RandomForestRegressor().fit(x_train, y_train)
print("Acurácia RFR:", round(rfr.score(x_test, y_test), 4) * 100)

print(process_time() - inicio)
