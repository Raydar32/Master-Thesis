# -*- coding: utf-8 -*-
"""
This script is an old script to reproduce experiments for the Thesis.
This particular script will reproduce results in form of a Ground Truth Table
using both Autoencoder and K-Means methods.

"""

from models.AutoEncoderEmbeddingClustering import AutoencoderEmbeddingClusteringModel
import pandas as pd
from models.KMeansClustering import KMeansClustering
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner
import time

# IP used as ground truth should be placed here:
groundTruth = {'ip1',
               'ip2',
               '...',

               }


def generateGTT(targets, labelset):
    """
        This method will build a GTT as you can find in the thesis.
    """

    temp = labelset.loc[labelset.index.isin(
        targets)]

    table = pd.crosstab(index=temp.index,
                        columns=temp["cluster"], margins=False)
    return table


# Loading dataset
DataC = ErgonDataCleaner()
DataC.setMinimumHoursToBeClustered(10)
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\aprile.csv")
DataC.cleanDataset()
DataC.setOutput("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
print("Ratio: ", DataC.getRatio())


# Loading and selecting 2 weeks of old traffic as base
df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
mask = (df['timestamp'] >= "2022-03-19 00:00:00+00:00") & (df['timestamp']
                                                           <= "2022-04-02 00:00:00+00:00")
df = df.loc[mask]


# =============================================================================
# Generating GTT for Autoencoder:
# Aggregation: 15 days @ 900s step
# Outlier Removal: Z-Score based
# Normalization: MinMaxScaler
# Average result: 0.87
# =============================================================================
time = "900s"
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
