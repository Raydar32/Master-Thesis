# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:41:39 2022

@author: Alessandro
"""

from models.ClusteringAlgorithmInterface import ClusteringAlgorithm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import numpy as np
import seaborn as sns


class SpectralClusteringModel(ClusteringAlgorithm):

    def clusterize(self):
        clustering = SpectralClustering(
            assign_labels='discretize').fit(self.df)
        self.final_score = silhouette_score(
            self.df, clustering.labels_)
        self.labeled_df = self.df.copy()
        self.labeled_df["cluster"] = clustering.labels_
        self.final_clusters = len(np.unique(clustering.labels_))
        return self.labeled_df

    def get_score(self):
        return self.final_score

    def get_c_num(self):
        return self.final_clusters

    def show_plot(self, title):

        transformed = self.pca_reduce_df(self.df, 2)
        plt.scatter(transformed[0], transformed[1],
                    c=self.labeled_df["cluster"], cmap='viridis')
        plt.title(title)
        plt.show()

    def pca_reduce_df(self, df, comps):
        pca = sklearnPCA(comps)  # 2-dimensional PCA
        transformed = pd.DataFrame(pca.fit_transform(df))
        return transformed
