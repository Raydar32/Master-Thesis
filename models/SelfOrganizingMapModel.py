# -*- coding: utf-8 -*-
"""
This script implements a SOM-based clustering methods based on the 
Sklearn documentation.
It will perform slightly better that other worst performing methods but 
it still gives us presentable and valid results.
"""
from models.ClusteringAlgorithmInterface import ClusteringAlgorithm
from minisom import MiniSom
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt


class SelfOrganizingMapModel(ClusteringAlgorithm):

    def clusterize(self):

        som = MiniSom(self.df.shape[1], self.df.shape[1],
                      self.df.shape[1], sigma=0.5, learning_rate=0.1)
        som.train_batch(self.df.values, 10000, verbose=self.verbose)
        winner_coordinates = np.array(
            [som.winner(x) for x in self.df.values]).T
        cluster_index = np.ravel_multi_index(
            winner_coordinates, (self.df.shape[1], self.df.shape[1]))

        self.final_clusters = len(np.unique(cluster_index))

        self.final_score = silhouette_score(
            self.df, cluster_index, metric='euclidean')
        self.labeled_df = self.df.copy()
        self.labeled_df["cluster"] = cluster_index

        return self.labeled_df

    def get_score(self):
        return self.final_score

    def get_c_num(self):
        return self.final_clusters

    def pca_reduce_df(self, df, comps):
        pca = sklearnPCA(comps)  # 2-dimensional PCA
        transformed = pd.DataFrame(pca.fit_transform(df))
        return transformed

    def show_plot(self, title):
        """
        This method perform a 2D plot using linear transformation to bring back
        dimensionality to 2.
        These plots will be reported in the thesis, the dimensionality reduction
        itself is done by pca_reduce_df.
        """
        transformed = self.pca_reduce_df(self.df, 2)
        plt.scatter(transformed[0], transformed[1],
                    c=self.labeled_df["cluster"], cmap='plasma', marker='.')
        plt.title(title)
        plt.show()
