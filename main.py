# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""



from ErgonDatasetCleaner import ErgonDataCleaner
import pandas as pd 
import numpy as np
from sklearn import preprocessing
#------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("C:\\Users\\Alessandro\\Downloads\\20.02to30.03_r.csv")
#DataC.cleanDataset()
#DataC.setOutput("5d_traffic_ergon_refined.csv")

print("Ratio: ", DataC.getRatio())


df = pd.read_csv("C:\\Users\\Alessandro\\Downloads\\20.02to30.03_r.csv")



#-------------------------- Estrazione di features -----------------------
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result



def extract_features(dataset):
    df_grouped = dataset.groupby("src_ip")
    extracted = pd.DataFrame()
    extracted["src_ip"] = df["src_ip"].unique().copy()
    extracted = extracted.set_index("src_ip")
    
    
    extracted = extracted.join(
                    df_grouped["dst_ip"].nunique()
                )
    
    extracted = extracted.join(
                    df_grouped["dst_port"].nunique()
                )
    extracted = extracted.join(
                    df_grouped["bytes_sent"].mean()
                )
    extracted = extracted.join(
                    df_grouped["bytes_recieved"].mean()      
                )

    extracted = extracted.join(
                    df_grouped["application"].nunique()
                )
    extracted = extracted.join(
                    df[df["dst_port"]<=1024].groupby("src_ip")["dst_port"]
                    .nunique()
                    .rename("small_ports")).replace(np.nan,0)
    

    extracted = extracted.join(
                    df[df["dst_port"]>1024].groupby("src_ip")["dst_port"]
                    .nunique()
                    .rename("big ports")).replace(np.nan,0)
    
    extracted = extracted.join(
                    df[df["transport"]=="tcp"].groupby("src_ip")["transport"]
                    .count()
                    .rename("tcp_num")).replace(np.nan,0)
    
    extracted = extracted.join(
                    df[df["transport"]=="udp"].groupby("src_ip")["transport"]
                    .count()
                    .rename("udp_num")).replace(np.nan,0)
    return extracted



extracted = pd.DataFrame()
df["timestamp"] = pd.to_datetime(df["timestamp"])
grouped = df.groupby(pd.Grouper(freq="1H", key="timestamp"))
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



##################### --------------------------- K-Means PoC ------------------------ ####################
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from kneed import KneeLocator

inertia = []
for candidate in range(2,30):
    
    km = KMeans(n_clusters=candidate,init="k-means++",n_init=1000)
    km.fit(extracted)
    labels = km.labels_
    score = silhouette_score(extracted, labels, metric='euclidean')
    inertia.append(km.inertia_)
    print("Optimizing ", candidate , " of ", 30," score ", score)



eps1 = KneeLocator(range(1, len(inertia)+1), inertia, curve='convex', direction='decreasing').knee
print("Epsilon ottimo: ", eps1)
print("Clusterizzazione finale K-means")
km = KMeans(n_clusters=eps1,init="k-means++",n_init=20000)
km.fit(extracted)
labels = km.labels_
score = silhouette_score(extracted, labels, metric='euclidean')
print("score: ", score)
extracted_kmeans = extracted.copy()
extracted_kmeans["c"] = km.labels_

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
for j in range(1,30):    
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










