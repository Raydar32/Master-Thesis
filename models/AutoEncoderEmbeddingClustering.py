# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:07:12 2022

@author: Alessandro
"""

from models.ClusteringAlgorithmInterface import ClusteringAlgorithm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from models.SpectralClusteringModel import SpectralClusteringModel
from keras import layers
from sklearn.model_selection import train_test_split
from models.KMeansClustering import KMeansClustering
from keras.models import Model
import tensorflow as tf
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from umap import UMAP
from models.DBScanClustering import DBScanClusteringModel
from models.SelfOrganizingMapModel import SelfOrganizingMapModel


class Autoencoder(Model):
    def __init__(self, bottleneck):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(512, activation='relu'),

            layers.Dense(bottleneck, activation='relu'),

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


class AutoencoderEmbeddingClusteringModel(ClusteringAlgorithm):

    def __init__(self, manifold, final_clustering):
        if manifold == None:

            self.final_clustering = final_clustering

            self.manifold = manifold
        if manifold != None:

            self.final_clustering = final_clustering

            self.manifold = manifold

    def clusterize(self):
        self.vprint("Creating train test subsets")
        df_train, df_test = train_test_split(self.df)
        self.vprint("Creating autoencoder model..")

        if self.manifold == None:
            bneck = 2
        else:
            bneck = 8
        autoencoder = Autoencoder(bneck)
        autoencoder.compile(optimizer='adam', loss="mse")
        autoencoder.fit(df_train, df_train,
                        epochs=100,
                        shuffle=False,
                        validation_data=(df_test, df_test), verbose=self.verbose)

        encoded_imgs = autoencoder.encoder(self.df.values).numpy()

        # ----- Clustering Using Strategy and no-Manifold ----

        # If IsoMap then apply transformation
        if self.manifold != None and self.manifold == "isomap":
            self.vprint("Applying isomap")
            encoded_imgs = Isomap(n_components=2).fit_transform(encoded_imgs)

        if self.manifold != None and self.manifold == "tsne":
            self.vprint("Applying T-Sne")
            encoded_imgs = TSNE(
                n_components=2, init='random').fit_transform(encoded_imgs)

        if self.manifold != None and self.manifold == "spectral":
            self.vprint("Applying spectral")
            encoded_imgs = SpectralEmbedding(
                n_components=2).fit_transform(encoded_imgs)

        if self.manifold != None and self.manifold == "umap":
            encoded_imgs = UMAP(
                n_components=2, init='random').fit_transform(encoded_imgs)

        # Then clustering.
        self.vprint("Clustering")
        df2 = pd.DataFrame(encoded_imgs)
        self.labeled_df = None
        if self.final_clustering == "kmeans":
            KMeans = KMeansClustering()
            KMeans.setVerbose(self.verbose)
            KMeans.setData(df2)

            self.labeled_df = KMeans.clusterize()
            self.final_score = KMeans.get_score()
            self.final_clusters = KMeans.get_c_num()

        if self.final_clustering == "spectral":
            spectralC = SpectralClusteringModel()
            spectralC.setVerbose(self.verbose)
            spectralC.setData(df2)
            self.labeled_df = spectralC.clusterize()
            self.final_score = spectralC.get_score()
            self.final_clusters = spectralC.get_c_num()

        if self.final_clustering == "dbscan":
            dbs = DBScanClusteringModel()
            dbs.setVerbose(self.verbose)
            dbs.setData(df2)
            self.labeled_df = dbs.clusterize()
            self.final_score = dbs.get_score()
            self.final_clusters = dbs.get_c_num()

        if self.final_clustering == "som":
            m = SelfOrganizingMapModel()
            m.setVerbose(self.verbose)
            m.setData(df2)
            self.labeled_df = m.clusterize()
            self.final_score = m.get_score()
            self.final_clusters = m.get_c_num()

        return self.labeled_df

    def get_score(self):
        return self.final_score

    def get_c_num(self):
        return self.final_clusters

    def show_plot(self, title):

        plt.scatter(self.labeled_df[0], self.labeled_df[1],
                    c=self.labeled_df["cluster"], cmap='plasma', marker='.')
        plt.title(title)
        plt.show()

    def pca_reduce_df(self, df, comps):
        pca = sklearnPCA(comps)  # 2-dimensional PCA
        transformed = pd.DataFrame(pca.fit_transform(df))
        return transformed
