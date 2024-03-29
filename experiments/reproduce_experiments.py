# -*- coding: utf-8 -*-
"""
This script is meant to reproduce experiments reported
in the slides in order to find optimal methods.
"""

from models.AutoEncoderEmbeddingClustering import AutoencoderEmbeddingClusteringModel
import pandas as pd
from models.KMeansClustering import KMeansClustering
from models.SpectralClusteringModel import SpectralClusteringModel
from models.SelfOrganizingMapModel import SelfOrganizingMapModel
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner
from models.DBScanClustering import DBScanClusteringModel
import time

# Cleaning dataset
DataC = ErgonDataCleaner()
DataC.setMinimumHoursToBeClustered(10)
DataC.setVerbose(True)
DataC.loadDataset("df_short.csv")
DataC.cleanDataset()
DataC.setOutput("df_short_r.csv")
print("Ratio: ", DataC.getRatio())
df = pd.read_csv("df_short_r.csv")


# Feature extraction
wcssRelevantFeatures = ['src_port',
                        'packets_dst_avg',
                        'packets_src_avg',
                        'dst_ip',
                        'dst_port',
                        'src_diversity',
                        'udp_ratio',
                        'tcp_ratio',
                        'dst_diversity',
                        'http_ratio',
                        'ssh_ratio',
                        'smtp_ratio', ]

unsup2supRelevantFeatures = ['packets_dst_avg',
                             'avg_duration',
                             'src_port',
                             'dst_diversity',
                             'dst_port',
                             'dst_ip',
                             'udp_ratio',
                             'tcp_ratio',
                             'http_ratio',
                             'src_diversity',
                             'ssh_ratio',
                             'smtp_ratio']


featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(unsup2supRelevantFeatures)
extracted = featureExtractor.createAggregatedFeatureSet(df, "600s")

featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(None)
extracted1 = featureExtractor.createAggregatedFeatureSet(df, "900s")


results = pd.DataFrame()


# Doing experiments
print("\033[H\033[J")
print("-----------------------------------------------------------------")
print("Batch di esperimenti, aggregazione: 1h, normalizzazione: max abs")
print("Pulizia ore > 10 , No-manifold embedded size: 2")
print("Manifold size 8")
print("-----------------------------------------------------------------")
for i in range(0, 10):
    print("--------------------------- ESPERIMENTO",
          i,  " ---------------------------")

    start_time = time.time()
    b = DBScanClusteringModel()
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    print("Classic clustering DBSCAN (no-opt) ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    b.show_plot("Classical DBSCAN clustering (no-opt) ")
    row = {"method": "Classic-DBSCAN", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    b = KMeansClustering()
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    print("Classic clustering beans (no-opt) ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    b.show_plot("Classical K-Means clustering (no-opt)")
    row = {"method": "Classic-KMeans", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = SpectralClusteringModel()
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    print("Classic clustering spectral (no-opt) ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    b.show_plot("Classical Spectral clustering (no-opt) ")
    row = {"method": "Classic-Spectral", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = SelfOrganizingMapModel()
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    print("Classic clustering SOM (no-opt) ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    b.show_plot("Classical SOM clustering (no-opt) ")
    row = {"method": "Classic-SOM", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("isomap", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder isomap - kmeans")
    print("Autoencoder clustering isomap - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "Isomap-KMEans", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("tsne", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  tsne - kmeans")
    print("Autoencoder clustering  tsne - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    start_time = time.time()
    row = {"method": "TSNE-KMEans", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    b = AutoencoderEmbeddingClusteringModel("spectral", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder spectral - kmeans")
    print("Autoencoder clustering spectral - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    start_time = time.time()
    row = {"method": "Spectral-KMEans", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    b = AutoencoderEmbeddingClusteringModel("umap", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  umap - kmeans")
    print("Autoencoder clustering  umap - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "UMAP-KMEans", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("isomap", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder isomap - dbscan")
    print("Autoencoder clustering isomap - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "Isomap-DBSCAN", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("tsne", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  tsne - dbscan")
    print("Autoencoder clustering  tsne - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "TSne-DBSCAN", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("spectral", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder spectral - dbscan")
    print("Autoencoder clustering spectral - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "Spectral-DBSCAN", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("umap", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  umap - dbscan")
    print("Autoencoder clustering  umap - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "UMAP-DBSCAN", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("isomap", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder isomap - Neural-SOM")
    print("Autoencoder clustering isomap - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "Isomap-SOM", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("tsne", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  tsne - Neural-SOM")
    print("Autoencoder clustering  tsne - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "TSne-SOM", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("spectral", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder spectral - Neural-SOM")
    print("Autoencoder clustering spectral - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "Spectral-SOM", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("umap", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  umap - Neural-SOM")
    print("Autoencoder clustering  umap - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    row = {"method": "UMAP-SOM", "expnum": i,
           "time": ((time.time() - start_time)), "clusters": b.get_c_num(), "score": b.get_score()}
    results = results.append(row, ignore_index=True)

    print("------------------------------------------------------------------------")


results.to_csv("experiment10.04.22.csv")
results.to_csv("J:\\experiment10.04.22.csv")
results = results.groupby("method").mean()
results.to_csv("experiment10.04.22_grouped.csv")
results.to_csv("J:\\experiment10.04.22_grouped.csv")
