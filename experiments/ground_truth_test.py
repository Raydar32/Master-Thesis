# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:48:28 2022

@author: Alessandro
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


def generateGTT(labelset):
    table = pd.crosstab(index=labelset.index,
                        columns=labelset["cluster"], margins=False)
    return table


# ------------------- Pulisco il dataset -------------------
# =============================================================================
DataC = ErgonDataCleaner()
DataC.setMinimumHoursToBeClustered(10)
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\aprile.csv")
DataC.cleanDataset()
DataC.setOutput("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
#
print("Ratio: ", DataC.getRatio())
# =============================================================================

# Scelgo il traffico di 15 giorni
df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
mask = (df['timestamp'] >= "2022-03-10 00:00:00+00:00") & (df['timestamp']
                                                           <= "2022-03-31 17:59:59+00:00")
df = df.loc[mask]

# ------------------- Feature extraction --------------------
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


time = "600s"
print("Currently exporting", time)
featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(unsup2supRelevantFeatures)
extracted = featureExtractor.createAggregatedFeatureSet(df, time)

b = KMeansClustering()
b.setVerbose(False)
b.setData(extracted)
labelset_kmeans_classic = b.clusterize()
print("Classic clustering beans (no-opt) ",
      b.get_c_num(), " ", b.get_score(), " ", )
b.show_plot("Classical K-Means clustering (no-opt)")

campione = {'192.168.121.47',
            '192.168.121.76',
            '192.168.121.80',
            '192.168.121.44',
            "192.168.121.11",  # Fine tecnici
            '192.168.121.6',
            '192.168.121.22',
            '192.168.121.21',
            '192.168.121.48',
            '192.168.121.45',  # Fine amministrativi
            }

selected_kmeans = labelset_kmeans_classic.loc[labelset_kmeans_classic.index.isin(
    campione)]

t_kmeans = generateGTT(selected_kmeans)

t_kmeans.to_csv("G:\\GitHubRepo\\Master-Thesis\\results\\" +
                str(time) + "_kmeans_30_gg.csv")
