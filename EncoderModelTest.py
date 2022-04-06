# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:12:20 2022

@author: Alessandro Mini
"""

import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from KMeansClustering import KMeansClustering
import numpy as np
import umap.umap_ as umap
from keras import regularizers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.preprocessing import sequence

import matplotlib.pyplot as plt


df = pd.read_csv("extracted_features.csv")
df = df.set_index("src_ip")
df_train, df_test = train_test_split(df)



inputs_dim = 13
encoder = Input(shape = (inputs_dim, ))
e = Dense(128, activation = "relu")(encoder)
e = Dropout(0.2)(e)
e = Dense(256, activation = "relu")(e)
e = Dropout(0.1)(e)
e = Dense(32, activation = "relu",activity_regularizer=l2(0.001))(e)

## bottleneck layer
n_bottleneck = 4
bottleneck_layer = "bottleneck_layer"
bottleneck = Dense(n_bottleneck, name = bottleneck_layer)(e)

## define the decoder (in reverse)
decoder = Dense(32, activation = "relu",activity_regularizer=l2(0.001))(bottleneck)
decoder = Dropout(0.1)(decoder)
decoder = Dense(256, activation = "relu")(decoder)
decoder = Dropout(0.2)(decoder)
decoder = Dense(128, activation = "relu")(decoder)


## output layer
output = Dense(inputs_dim, activation = "sigmoid")(decoder)
## model
model = Model(inputs = encoder, outputs = output)

encoder = Model(inputs = model.input, outputs = bottleneck)
model.compile(loss = "mse", optimizer = "adam")
history = model.fit(
    df_train,
    df_train,
    shuffle = False,
    batch_size = 128,
    epochs = 100,
    verbose = 1,
    validation_data = (df_test, df_test)
)





encoded_imgs = encoder.predict(df)

from sklearn.manifold import Isomap
encoded_imgs = Isomap(n_components=2).fit_transform(encoded_imgs)


df2 = pd.DataFrame(encoded_imgs)
KMeans = KMeansClustering()
KMeans.setVerbose(True)
KMeans.setData(df2)
labelset = KMeans.clusterize()
print("Score encoded: ", KMeans.final_score)
KMeans.show_plot()
df["cluster"] = labelset["cluster"].values
mio = df.loc[df.index=="192.168.121.47"]
amm = df.loc[df.index=="192.168.121.48"]

plt.scatter(labelset[0], labelset[1], c=labelset["cluster"],cmap='viridis')
