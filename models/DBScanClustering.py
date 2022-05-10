# -*- coding: utf-8 -*-
"""
This script implements the DBSCAN Clustering algorithm with
Knn-distances eblow method to determine epsilon and nn parameters.
This script here is just experimental as the others, it will not produce
great results with the input data due to their nature.
"""


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from kneed import KneeLocator
from models.ClusteringAlgorithmInterface import ClusteringAlgorithm
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import numpy as np


class DBScanClusteringModel(ClusteringAlgorithm):

    def clusterize(self):
        """
        Override of the clusterize method.
        """
        nn = 16
        distances, indices = NearestNeighbors(
            n_neighbors=nn).fit(self.df).kneighbors(self.df)
        distances = np.sort(distances, axis=0)[:, 1]

        distance_index = KneeLocator(range(1, len(distances)+1), distances,
                                     curve='convex', direction='increasing').knee
        epsilon = distances[distance_index]
        clustering = DBSCAN(eps=epsilon, min_samples=nn).fit(self.df)
        clustering.labels_
        self.final_score = silhouette_score(
            self.df, clustering.labels_, metric='euclidean')
        self.labeled_df = self.df.copy()
        self.labeled_df["cluster"] = clustering.labels_
        self.final_clusters = len(np.unique(clustering.labels_))
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
        transformed = self.pca_reduce_df(self.df, 2)
        plt.scatter(transformed[0], transformed[1],
                    c=self.labeled_df["cluster"], cmap='plasma', marker='.')
        plt.title(title)
        plt.show()
