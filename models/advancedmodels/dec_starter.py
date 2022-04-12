# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:50:38 2022

@author: Alessandro Mini
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras_dec import DEC
from tensorflow.keras.optimizers import SGD
from keras.initializers import VarianceScaling, GlorotUniform
from sklearn.metrics import silhouette_score

df = pd.read_csv("features.csv")
df = df.set_index("src_ip")
df = df.values/1.0
df_train, df_test = train_test_split(df)
pretrain_optimizer = SGD(lr=1, momentum=0.9)
file = 'G:\\GitHubRepo\\Master-Thesis\\test\\pretrain_log.csv'

init = GlorotUniform()


dec = DEC(dims=[df_train.shape[-1], 512, 128, 8],  init=init)


dec.pretrain(x=df, optimizer=pretrain_optimizer,
             epochs=50,
             save_dir='G:\\GitHubRepo\\Master-Thesis\\test\\',
             )

dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')


dec.fit(df, update_interval=150, batch_size=64,
        save_dir='G:\\GitHubRepo\\Master-Thesis\\test\\')


k = dec.predict(df)
sc = silhouette_score(df, k)
