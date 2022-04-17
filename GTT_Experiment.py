# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:06:52 2022

@author: Alessandro

This script here should be the main script of 
Firewall profiler tool for my Master Thesis @ University of Florence.

"""

from models.AutoEncoderEmbeddingClustering import AutoencoderEmbeddingClusteringModel
import pandas as pd
from models.KMeansClustering import KMeansClustering
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner
import time


groundTruth = {'192.168.121.47',
               '192.168.121.76',
               '192.168.121.80',
               '192.168.121.6',
               '192.168.121.21',
               '192.168.121.45',
               }


def generateGTT(targets, labelset):

    temp = labelset.loc[labelset.index.isin(
        targets)]

    table = pd.crosstab(index=temp.index,
                        columns=temp["cluster"], margins=False)
    return table


# # â =============================================================================
DataC = ErgonDataCleaner()
DataC.setMinimumHoursToBeClustered(10)
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\aprile.csv")
DataC.cleanDataset()
DataC.setOutput("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
print("Ratio: ", DataC.getRatio())
# #
# # =============================================================================

# Loading and selecting 2 weeks of old traffic as base
df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")


# =============================================================================
df['timestamp'] = pd.to_datetime(df['timestamp'])
mask = (df['timestamp'] >= "2022-03-19 00:00:00+00:00") & (df['timestamp']
                                                           <= "2022-04-02 00:00:00+00:00")
df = df.loc[mask]
# =============================================================================


# =============================================================================
# Generating GTT for Autoencoder:
# Aggregation: 15 days @ 900s step
# Outlier Removal: Z-Score based
# Normalization: MinMaxScaler
# Average result: 0.87
# =============================================================================
time = "600s"
print("Processing", time)
featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(None)
extractedForAutoencoder = featureExtractor.createAggregatedFeatureSet(df, time)

b = AutoencoderEmbeddingClusteringModel("isomap", "kmeans")
b.setVerbose(True)
b.setData(extractedForAutoencoder)
labelset_autoencoder = b.clusterize()
b.show_plot("Autoencoder isomap - kmeans")
print("Autoencoder clustering isomap - kmeans ",
      b.get_c_num(), " ", b.get_score(), " ",)

# Generating GTT for autoencoder
autoencoderGTT = generateGTT(groundTruth, labelset_autoencoder)


# =============================================================================
# Generating GTT for KMeans With Unsup2Sup optimization:
# Aggregation: 15 days @ 600s step
# Outlier Removal: Z-Score based
# Normalization: MinMaxScaler
# Average result: 0.92
# =============================================================================


unsup2supUnrelevantFeatures = ['packets_dst_avg',
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
featureExtractor.setEclusionList(unsup2supUnrelevantFeatures)
extracted = featureExtractor.createAggregatedFeatureSet(df, "600s")


b = KMeansClustering()
b.setVerbose(False)
b.setData(extracted)
labelset_kmeans_classic = b.clusterize()
print("Classic clustering beans (no-opt) ",
      b.get_c_num(), " ", b.get_score(), " ", )
b.show_plot("Classical K-Means clustering (no-opt)")

KMeansClassicGTT = generateGTT(groundTruth, labelset_kmeans_classic)
