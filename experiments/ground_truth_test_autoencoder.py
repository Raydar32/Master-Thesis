

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


print("Script started")
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


time = "900s"
print("Processing", time)
featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(None)
extracted1 = featureExtractor.createAggregatedFeatureSet(df, time)

# Clustering autoencoder 1
b = AutoencoderEmbeddingClusteringModel("isomap", "kmeans")
b.setVerbose(False)
b.setData(extracted1)
labelset_autoencoder = b.clusterize()
b.show_plot("Autoencoder isomap - kmeans")
print("Autoencoder clustering isomap - kmeans ",
      b.get_c_num(), " ", b.get_score(), " ",)
labelset_autoencoder["src_ip"] = extracted1.index.values
labelset_autoencoder = labelset_autoencoder.set_index("src_ip")

selected_autoencoder = labelset_autoencoder.loc[labelset_autoencoder.index.isin(
    campione)]

t_autoencoder = generateGTT(selected_autoencoder)

t_autoencoder.to_csv("G:\\GitHubRepo\\Master-Thesis\\results\\" +
                     str(time) + "isomap_kmeans30gg.csv")
