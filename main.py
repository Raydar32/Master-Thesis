# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""


from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import pandas as pd
from models.KMeansClustering import KMeansClustering
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner

# ------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
# DataC.cleanDataset()
DataC.setOutput("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")

print("Ratio: ", DataC.getRatio())


df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")


wcssRelevantFeatures = ['packets_dst_avg', 'avg_duration', 'src_port', 'dst_diversity', 'dst_port',
                        'dst_ip', 'udp_ratio', 'tcp_ratio', 'http_ratio', 'src_diversity', 'ssh_ratio', 'smtp_ratio']


featureExtractor = PaloAltoFeatureExtractor()
featureExtractor.setEclusionList(None)
extracted = featureExtractor.createAggregatedFeatureSet(df, "1800s")


KMeans = KMeansClustering()
KMeans.setVerbose(True)
KMeans.setData(extracted)
labeled = KMeans.clusterize()
print("Clustering k-means: ", KMeans.get_c_num(), " ", KMeans.get_score())
KMeans.show_plot()


#clients =  extracted[extracted.index.str.contains("192.168.121")]
#mgmt =  extracted[extracted.index.str.contains("192.168.111")]
#servers_firenze =  extracted[extracted.index.str.contains("192.168.111")]
#servers_siena =  extracted[extracted.index.str.contains("192.168.6")]
#voip =  extracted[extracted.index.str.contains("192.168.111")]
#mio = extracted.loc[extracted.index=="192.168.121.47"]


# ---------------------------------- Spectral  POC -------------------------------------------------


# Clustering spettrale
clustering = SpectralClustering(
    n_clusters=2, metric='nearest_neighbors', eigen_solver='amg').fit(extracted)
labels_spectral = clustering.labels_
print("SC spectral: ", silhouette_score(
    extracted, labels_spectral, metric="euclidean"))
print("DB spectral: ", davies_bouldin_score(extracted, labels_spectral))

# Clustering agglomerativo
clustering = AgglomerativeClustering(n_clusters=6).fit(extracted)
labels_agglomerative = clustering.labels_
print("SC agg: ", silhouette_score(
    extracted, labels_agglomerative, metric='euclidean'))
print("DB agg: ", davies_bouldin_score(extracted, labels_agglomerative))


# Birch
brc = Birch(n_clusters=6)
brc.fit_predict(extracted)
birch_labels = brc.labels_
print("SC bir: ", silhouette_score(
    extracted, birch_labels, metric='nearest_neighbors',))
print("DB bir: ", davies_bouldin_score(extracted, birch_labels))
