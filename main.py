# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""



from ErgonDatasetCleaner import ErgonDataCleaner
import pandas as pd 
import numpy as np
from ErgonFeaturesExtractor import extract_features
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from KMeansClustering import KMeansClustering
from scipy import stats
from sklearn.preprocessing import MaxAbsScaler
#------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")
#DataC.cleanDataset()
DataC.setOutput("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")

print("Ratio: ", DataC.getRatio())


df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\aprile_r.csv")



def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result



        



#-------------------------- Estrazione di features -----------------------
extracted = pd.DataFrame()
df["timestamp"] = pd.to_datetime(df["timestamp"])
grouped = df.groupby(pd.Grouper(freq="1h", key="timestamp"))
i = 0
for hour in grouped.groups.keys():
    esito = False
    try:
        a = grouped.get_group(hour)
        esito = True
    except:
        esito = False
        pass
    else:
        esito = True
        extracted = extracted.append(extract_features(a))
    i = i + 1 
    print("Scanning time-slot :",i," di ", len(grouped.groups.keys()),hour.hour, ":",hour.minute," giorno ", hour.day, "//",
          hour.month, " totale ", len(grouped.groups.keys()), esito)

#extracted = normalize(extracted)
extracted_values = MaxAbsScaler().fit_transform(extracted)
extracted = pd.DataFrame(extracted_values,index=extracted.index,columns=extracted.columns)
extracted=extracted[(np.abs(stats.zscore(extracted)) < 3).all(axis=1)]

#----------- Test: rimozione featues non rilevanti -------------
#+del extracted["http_ratio"]
#del extracted["dst_ip"]
#del extracted["bytes"]
#del extracted["packets"]
#del extracted["dst_port"]
#del extracted["ssh_ratio"]
#del extracted["avg_duration"]
#del extracted["smtp_ratio"]




KMeans = KMeansClustering()
KMeans.setVerbose(True)
KMeans.setData(extracted)
labeled = KMeans.clusterize()
score = KMeans.get_score()
cnum = KMeans.get_c_num()
print("Clustering k-means: " , cnum, " ",score)



#clients =  extracted[extracted.index.str.contains("192.168.121")]
#mgmt =  extracted[extracted.index.str.contains("192.168.111")]
#servers_firenze =  extracted[extracted.index.str.contains("192.168.111")]
#servers_siena =  extracted[extracted.index.str.contains("192.168.6")]
#voip =  extracted[extracted.index.str.contains("192.168.111")]
#mio = extracted.loc[extracted.index=="192.168.121.47"]


#---------------------------------- Spectral  POC -------------------------------------------------



#Clustering spettrale
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(assign_labels='discretize').fit(extracted)
labels_spectral = clustering.labels_
print("SC spectral: ", silhouette_score(extracted, labels_spectral,metric="euclidean"))
print("DB spectral: ", davies_bouldin_score(extracted, labels_spectral))

#Clustering agglomerativo
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=6).fit(extracted)
labels_agglomerative = clustering.labels_
print("SC agg: ", silhouette_score(extracted, labels_agglomerative, metric='euclidean'))
print("DB agg: ", davies_bouldin_score(extracted, labels_agglomerative))



#Birch
from sklearn.cluster import Birch
brc = Birch(n_clusters=6)
brc.fit_predict(extracted)
birch_labels = brc.labels_
print("SC bir: ", silhouette_score(extracted, birch_labels, metric='euclidean'))
print("DB bir: ", davies_bouldin_score(extracted, birch_labels))

















