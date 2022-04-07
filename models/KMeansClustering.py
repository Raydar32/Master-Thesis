# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:46:48 2022

@author: Alessandro Mini
"""
from models.ClusteringAlgorithmInterface import ClusteringAlgorithm
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt


class KMeansClustering(ClusteringAlgorithm):

    def optimize(self):
        inertia = []
        for candidate in range(2, 30):
            self.vprint("it: ", candidate)
            km = KMeans(n_clusters=candidate, init="k-means++")
            km.fit(self.df)
            inertia.append(km.inertia_)
        eps1 = KneeLocator(range(1, len(inertia)+1), inertia,
                           curve='convex', direction='decreasing').knee
        return eps1

    def clusterize(self):
        eps_opt = self.optimize()
        self.final_clusters = eps_opt
        km = KMeans(n_clusters=eps_opt, init="k-means++")
        km.fit(self.df)
        self.final_score = silhouette_score(
            self.df, km.labels_, metric='euclidean')
        self.labeled_df = self.df.copy()
        self.labeled_df["cluster"] = km.labels_
        return self.labeled_df

    def get_score(self):
        return self.final_score

    def get_c_num(self):
        return self.final_clusters

    def pca_reduce_df(self, df, comps):
        pca = sklearnPCA(comps)  # 2-dimensional PCA
        transformed = pd.DataFrame(pca.fit_transform(df))
        return transformed

    def show_plot(self):
        self.vprint("Reducing dimensionality using PCA")
        transformed = self.pca_reduce_df(self.df, 2)
        plt.scatter(transformed[0], transformed[1],
                    c=self.labeled_df["cluster"], cmap='viridis')
