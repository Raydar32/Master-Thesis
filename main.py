# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""


from models.AutoEncoderEmbeddingClustering import AutoencoderEmbeddingClusteringModel
import pandas as pd
from models.KMeansClustering import KMeansClustering
from models.SpectralClusteringModel import SpectralClusteringModel
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner

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

print("Metodi classici")

km = KMeansClustering()
km.setVerbose(False)
km.setData(extracted)
km.clusterize()
print("Classic clustering kmeans (no-opt) ",
      km.get_c_num(), " ", km.get_score())
km.show_plot("Classical K-Means clustering (no-opt)")


spec = SpectralClusteringModel()
spec.setVerbose(False)
spec.setData(extracted)
spec.clusterize()
print("Classic clustering spectral (no-opt) ",
      spec.get_c_num(), " ", spec.get_score())
spec.show_plot("Classical Spectral clustering (no-opt) ")


print("Metodi deep learning ")
print("Working: isomap - kmeans")
b = AutoencoderEmbeddingClusteringModel("isomap", "kmeans")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder isomap - kmeans")
print("Autoencoder clustering isomap - kmeans ",
      b.get_c_num(), " ", b.get_score())

print("Working: isomap - spectral")
b = AutoencoderEmbeddingClusteringModel("isomap", "spectral")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder isomap - spectral")
print("Autoencoder clustering isomap - spectral ",
      b.get_c_num(), " ", b.get_score())

print("Working: tsne - kmeans")
b = AutoencoderEmbeddingClusteringModel("tsne", "kmeans")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder  tsne - kmeans")
print("Autoencoder clustering  tsne - kmeans ",
      b.get_c_num(), " ", b.get_score())

print("Working: tsne - spectral")
b = AutoencoderEmbeddingClusteringModel("tsne", "spectral")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder tsne - spectral")
print("Autoencoder clustering tsne - spectral ",
      b.get_c_num(), " ", b.get_score())

print("Working: spectral - kmeans")
b = AutoencoderEmbeddingClusteringModel("spectral", "kmeans")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder spectral - kmeans")
print("Autoencoder clustering spectral - kmeans ",
      b.get_c_num(), " ", b.get_score())


print("Working: spectral - spectral")
b = AutoencoderEmbeddingClusteringModel("spectral", "spectral")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder spectral - spectral")
print("Autoencoder clustering spectral - spectral",
      b.get_c_num(), " ", b.get_score())

print("Working: umap - kmeans")
b = AutoencoderEmbeddingClusteringModel("umap", "kmeans")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder  umap - kmeans")
print("Autoencoder clustering  umap - kmeans ",
      b.get_c_num(), " ", b.get_score())

print("Working: umap - spectral")
b = AutoencoderEmbeddingClusteringModel("umap", "spectral")
b.setVerbose(False)
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder umap - spectral")
print("Autoencoder clustering umap - spectral ",
      b.get_c_num(), " ", b.get_score())
b.setData(extracted)
b.clusterize()
b.show_plot("Autoencoder umap - spectral")
print("Autoencoder clustering umap - spectral ",
      b.get_c_num(), " ", b.get_score())
