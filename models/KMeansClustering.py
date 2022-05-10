# -*- coding: utf-8 -*-
"""
This script is one of the most important of the whole project, it implements
K-Means, one of the "basic" clustering algorithm that will be used for both 
classic and neural models.
This strategy will be based on:
    
    -KMeans++ for weight inizialization.
    -Elbow method over inertia (WCSS) to determine (automatically) the optimal
     value for K.
    -Metric will be based on Sihlouette score.
    
For further information look for the thesis.pdf.

"""
from models.ClusteringAlgorithmInterface import ClusteringAlgorithm
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import numpy as np


class KMeansClustering(ClusteringAlgorithm):
    """
    Main method that implements the ClusteringAlgorithm interface.
    """

    def optimize(self):
        """
        This method will perform the Inertia (or WCSS) elbow method using
        kneed library.
        It will span for 30 candidates.
        """
        inertia = []
        for candidate in range(2, 30):
            self.vprint("Optimizing K-Means it: ", candidate)
            km = KMeans(n_clusters=candidate, init="k-means++")
            km.fit(self.df)
            inertia.append(km.inertia_)
        eps1 = KneeLocator(range(1, len(inertia)+1), inertia,
                           curve='convex', direction='decreasing').knee
        return eps1

    def clusterize(self):
        """
        This method performs the basic clustering operation.

        """
        eps_opt = self.optimize()
        self.final_clusters = eps_opt
        km = KMeans(n_clusters=eps_opt, init="k-means++")
        km.fit(self.df)
        self.final_score = silhouette_score(
            self.df, km.labels_, metric='euclidean')
        self.labeled_df = self.df.copy()
        self.labeled_df["cluster"] = km.labels_
        self.final_clusters = len(np.unique(km.labels_))
        return self.labeled_df

    def get_score(self):
        return self.final_score

    def get_c_num(self):
        return self.final_clusters

    def pca_reduce_df(self, df, comps):
        pca = sklearnPCA(comps)
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
