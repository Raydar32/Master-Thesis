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

from keras.models import Model
from keras.layers import Dense, Input
from keras.preprocessing import sequence




df = pd.read_csv("extracted_feature_set.csv")
df = df.set_index("src_ip")
df_train, df_test = train_test_split(df)


inputs_dim = 13
encoder = Input(shape = (inputs_dim, ))
e = Dense(512)(encoder)
e = Dense(512, activation = "sigmoid")(e)
e = Dense(128)(e)
e = Dense(128, activation = "sigmoid")(e)
e = Dense(32)(e)
e = Dense(32)(e)

## bottleneck layer
n_bottleneck = 6
## defining it with a name to extract it later
bottleneck_layer = "bottleneck_layer"
# can also be defined with an activation function, relu for instance
bottleneck = Dense(n_bottleneck, name = bottleneck_layer)(e)
## define the decoder (in reverse)
decoder = Dense(32)(bottleneck)
decoder = Dense(32)(decoder)
decoder = Dense(128, activation = "sigmoid")(decoder)
decoder = Dense(128)(decoder)
decoder = Dense(512, activation = "sigmoid")(decoder)
decoder = Dense(512)(decoder)

## output layer
output = Dense(inputs_dim)(decoder)
## model
model = Model(inputs = encoder, outputs = output)

encoder = Model(inputs = model.input, outputs = bottleneck)
model.compile(loss = "mse", optimizer = "adam")
history = model.fit(
    df_train,
    df_train,
    shuffle = False,
    epochs = 15,
    verbose = 1,
    validation_data = (df_test, df_test)
)





encoded_imgs = encoder.predict(df)
#decoded_imgs = decoder.predict(encoded_imgs)


encoded_imgs = umap.UMAP(n_components =  2, 
                           metric = "euclidean",
                          
                                         
                          ).fit_transform(encoded_imgs)



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



from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
scores = {0:0}
for j in range(1,35):    
    nn = df2.shape[1]*j
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(df2)
    distances, indices = nbrs.kneighbors(df2)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]    
    knee_x = KneeLocator(range(1, len(distances)+1), distances, curve='convex', direction='increasing').knee
    eps_ = distances[knee_x]    
    clustering = DBSCAN(eps=eps_, min_samples=nn).fit(df2)
    score = silhouette_score(df2, clustering.labels_, metric='euclidean')
    print("nn: ", nn, " score ", score , " nc ", max(clustering.labels_))
    scores[j] =score
    
#Adesso applico il clustering con il massimo score
max_score = max(scores, key=scores.get)
right_eps = scores[max_score]
clustering = DBSCAN(eps=right_eps, min_samples=max_score*df2.shape[1]).fit(df2)
score = silhouette_score(df2, clustering.labels_, metric='euclidean')
print("Score finale: ", score)


