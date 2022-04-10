# -*- coding: utf-8 -*-
"""
Test file to keep everything together
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
# ------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setMinimumHoursToBeClustered(10)
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
# DataC.cleanDataset()
DataC.setOutput("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")

print("Ratio: ", DataC.getRatio())


df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")


# ------------------- Feature extraction --------------------
wcssRelevantFeatures = ['packets_dst_avg', 'avg_duration', 'src_port', 'dst_diversity', 'dst_port',
                        'dst_ip', 'udp_ratio', 'tcp_ratio', 'http_ratio', 'src_diversity', 'ssh_ratio', 'smtp_ratio']


featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(None)
extracted = featureExtractor.createAggregatedFeatureSet(df, "1h")


# -------------------- Clustering Experiments -----------------
print("\033[H\033[J")
print("-----------------------------------------------------------------")
print("Batch di esperimenti, aggregazione: 1h, normalizzazione: max abs")
print("Pulizia ore > 10 , No-manifold embedded size: 2")
print("Manifold size 8")
print("-----------------------------------------------------------------")
for i in range(0, 10):
    print("--------------------------- ESPERIMENTO",
          i,  " ---------------------------")

    # Modelli base (Senza ottimizzazione features)
    start_time = time.time()
    dbs = DBScanClusteringModel()
    dbs.setVerbose(False)
    dbs.setData(extracted)
    dbs.clusterize()
    print("Classic clustering DBSCAN (no-opt) ",
          dbs.get_c_num(), " ", dbs.get_score(), " ", ((time.time() - start_time)), "s")
    dbs.show_plot("Classical DBSCAN clustering (no-opt) ")

    start_time = time.time()
    km = KMeansClustering()
    km.setVerbose(False)
    km.setData(extracted)
    km.clusterize()
    print("Classic clustering kmeans (no-opt) ",
          km.get_c_num(), " ", km.get_score(), " ", ((time.time() - start_time)), "s")
    km.show_plot("Classical K-Means clustering (no-opt)")

    start_time = time.time()
    spec = SpectralClusteringModel()
    spec.setVerbose(False)
    spec.setData(extracted)
    spec.clusterize()
    print("Classic clustering spectral (no-opt) ",
          spec.get_c_num(), " ", spec.get_score(), " ", ((time.time() - start_time)), "s")
    spec.show_plot("Classical Spectral clustering (no-opt) ")

    start_time = time.time()
    m = SelfOrganizingMapModel()
    m.setVerbose(False)
    m.setData(extracted)
    m.clusterize()
    print("Classic clustering SOM (no-opt) ",
          m.get_c_num(), " ", m.get_score(), " ", ((time.time() - start_time)), "s")
    m.show_plot("Classical SOM clustering (no-opt) ")

    # Autoencoder senza niente

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel(None, "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder base  k-means")
    print("Autoencoder base  k-means ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel(None, "spectral")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder base spectral")
    print("Autoencoder base spectral ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel(None, "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder base som")
    print("Autoencoder base som ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel(None, "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  base dbscan")
    print("Autoencoder base dbscan",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    # Modelli deep learning - Kmeans con vari Manifold
    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("isomap", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder isomap - kmeans")
    print("Autoencoder clustering isomap - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("tsne", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  tsne - kmeans")
    print("Autoencoder clustering  tsne - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    start_time = time.time()

    b = AutoencoderEmbeddingClusteringModel("spectral", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder spectral - kmeans")
    print("Autoencoder clustering spectral - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
    start_time = time.time()

    b = AutoencoderEmbeddingClusteringModel("umap", "kmeans")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  umap - kmeans")
    print("Autoencoder clustering  umap - kmeans ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    # Modelli deep learning - DBSCAN con vari Manifold
    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("isomap", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder isomap - dbscan")
    print("Autoencoder clustering isomap - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("tsne", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  tsne - dbscan")
    print("Autoencoder clustering  tsne - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("spectral", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder spectral - dbscan")
    print("Autoencoder clustering spectral - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("umap", "dbscan")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  umap - dbscan")
    print("Autoencoder clustering  umap - dbscan ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    # Modelli deep learning - SOM con vari Manifold
    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("isomap", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder isomap - Neural-SOM")
    print("Autoencoder clustering isomap - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("tsne", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  tsne - Neural-SOM")
    print("Autoencoder clustering  tsne - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("spectral", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder spectral - Neural-SOM")
    print("Autoencoder clustering spectral - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")

    start_time = time.time()
    b = AutoencoderEmbeddingClusteringModel("umap", "som")
    b.setVerbose(False)
    b.setData(extracted)
    b.clusterize()
    b.show_plot("Autoencoder  umap - Neural-SOM")
    print("Autoencoder clustering  umap - Neural-SOM ",
          b.get_c_num(), " ", b.get_score(), " ", ((time.time() - start_time)), "s")
