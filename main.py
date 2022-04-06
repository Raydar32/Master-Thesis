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

#------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("C:\\InProgress\\Tesi\\aprile.csv")
DataC.cleanDataset()
DataC.setOutput("C:\\InProgress\\Tesi\\aprile_r.csv")

print("Ratio: ", DataC.getRatio())


df = pd.read_csv("C:\\InProgress\\Tesi\\aprile_r.csv")



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

extracted = normalize(extracted)


#----------- Test: rimozione featues non rilevanti -------------
del extracted["http_ratio"]
del extracted["dst_ip"]
del extracted["bytes"]
del extracted["packets"]
del extracted["dst_port"]
del extracted["ssh_ratio"]
del extracted["avg_duration"]
del extracted["smtp_ratio"]




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


#---------------------------------- DBSCAN POC -------------------------------------------------


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN




scores = {0:0}
for j in range(1,15):    
    nn = extracted.shape[1]*j
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(extracted)
    distances, indices = nbrs.kneighbors(extracted)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]    
    knee_x = KneeLocator(range(1, len(distances)+1), distances, curve='convex', direction='increasing').knee
    eps_ = distances[knee_x]    
    clustering = DBSCAN(eps=eps_, min_samples=nn).fit(extracted)
    score = silhouette_score(extracted, clustering.labels_, metric='euclidean')
    print("nn: ", nn, " score ", score , " nc ", max(clustering.labels_))
    scores[j] =score
    
#Adesso applico il clustering con il massimo score
max_score = max(scores, key=scores.get)
right_eps = scores[max_score]
clustering = DBSCAN(eps=right_eps, min_samples=max_score*extracted.shape[1]).fit(extracted)
score = silhouette_score(extracted, clustering.labels_, metric='euclidean')
print("Score finale: ", score)
extracted_dbscan = extracted.copy()
extracted_dbscan["c"] = np.nan 
extracted_dbscan["c"] = clustering.labels_

#ward poc










