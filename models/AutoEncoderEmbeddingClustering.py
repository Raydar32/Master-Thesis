# -*- coding: utf-8 -*-
"""
This script implements an Autoencoder method for clustering based on the
N2D paper.
This is one of the most important scripts of the whole project because this 
method, opportunely tuned, can produce impressing results considering the 
fully-unsupervised nature of the algorithm.

All the methods and procedures here are to determine the best fitting model, 
that will be Autoencoder-Kmeans with Isomap manifold.

Unused code may be removed in a further implementation.
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
    """
    this method implements an autoencoder using Keras documentation.
    The dimensionality of the layers of the autoencoder has been determined
    with experiemnts.
    Most performing size is the one that is implemented below, we can achieve
    a reconstrcution loss approx 10^-6/-7.
    """

    def __init__(self, bottleneck):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(bottleneck, activation='relu'),

        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(18, activation='sigmoid'),

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
            bneck = 4
        else:
            bneck = 8

        self.vprint("Training autoencoder")
        autoencoder = Autoencoder(bneck)
        autoencoder.compile(optimizer='adam', loss="mse")
        autoencoder.fit(df_train, df_train,
                        epochs=100,
                        shuffle=False,
                        validation_data=(df_test, df_test), verbose=self.verbose)

        self.vprint("Applying encoder to input")
        encoded_imgs = autoencoder.encoder(self.df.values).numpy()

        # ----- Clustering Using Strategy and no-Manifold ----

        # If IsoMap then apply transformation
        if self.manifold != None and self.manifold == "isomap":
            self.vprint("Applying isomap")
            encoded_imgs = Isomap(
                n_components=3).fit_transform(encoded_imgs)

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

        self.labeled_df["src_ip"] = self.df.index.values
        self.labeled_df = self.labeled_df.set_index("src_ip")

        return self.labeled_df

    def get_score(self):
        return self.final_score

    def get_c_num(self):
        return self.final_clusters

    def show_plot(self, title):
        """
        This method perform a 2D plot using linear transformation to bring back
        dimensionality to 2.
        These plots will be reported in the thesis, the dimensionality reduction
        itself is done by pca_reduce_df.
        """
        plt.scatter(self.labeled_df[0], self.labeled_df[1],
                    c=self.labeled_df["cluster"], cmap='plasma', marker='.')
        plt.title(title)
        plt.show()

    def pca_reduce_df(self, df, comps):
        pca = sklearnPCA(comps)  # 2-dimensional PCA for plot
        transformed = pd.DataFrame(pca.fit_transform(df))
        return transformed
        return transformed
