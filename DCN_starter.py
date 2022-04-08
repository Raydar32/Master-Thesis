# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:50:50 2022

@author: Alessandro Mini
"""



import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import tensorflow as tf
from sklearn.manifold import Isomap
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from models.KMeansClustering import KMeansClustering
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append('..')
from DCN import DCN
import dcn_callbacks as dcn_call

#Author: stallmo


dcn_model = DCN(latent_dim=8,
                input_dim=17,
                n_clusters=7,
                lamda=200)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

dcn_model.compile(run_eagerly=True, 
                  optimizer=optimizer)


df = pd.read_csv("features.csv")
df = df.set_index("src_ip")
df = df.values/1.0
df_train, df_test = train_test_split(df)
clust_learn_updater = dcn_call.ClustLearningRateUpdater() 

dcn_model.pretrain(df_train, 
                   batch_size = 4096,
                   epochs=100,                
                   verbose=True,
               )
    
dcn_model.fit(df_train, df_train, 
              shuffle=False,
              batch_size = 4096,
              epochs=100,
              callbacks=[clust_learn_updater
                         ])

latent_x = dcn_model.encoder(df) 
# ... and the the assignments
deep_assignments = dcn_model.get_assignment(latent_x)

from sklearn.metrics import silhouette_score
sc = silhouette_score(df,dcn_model.get_assignment(latent_x),metric="euclidean")
print(sc)