# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:12:20 2022

@author: Alessandro Mini
"""

from sklearn.manifold import Isomap
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from models.KMeansClustering import KMeansClustering
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt


df = pd.read_csv("features.csv")
df = df.set_index("src_ip")
df_train, df_test = train_test_split(df)


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(512, activation='relu'),

            layers.Dense(8, activation='relu'),

        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(17, activation='sigmoid'),

        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss="mse")
autoencoder.fit(df_train, df_train,
                epochs=100,
                shuffle=False,
                validation_data=(df_test, df_test))


encoded_imgs = autoencoder.encoder(df.values)
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


encoded_imgs = Isomap(n_components=2).fit_transform(encoded_imgs)


df2 = pd.DataFrame(encoded_imgs)
KMeans = KMeansClustering()
KMeans.setVerbose(True)
KMeans.setData(df2)
labelset = KMeans.clusterize()
print("Score encoded: ", KMeans.final_score)
KMeans.show_plot()
df["cluster"] = labelset["cluster"].values
mio = df.loc[df.index == "192.168.121.47"]
amm = df.loc[df.index == "192.168.121.48"]

plt.scatter(labelset[0], labelset[1], c=labelset["cluster"], cmap='viridis')
