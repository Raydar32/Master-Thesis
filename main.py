# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""



from ErgonDatasetCleaner import ErgonDataCleaner
import pandas as pd 
#------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("G:\\30d_traffic_ergon.csv")
DataC.cleanDataset()
DataC.setOutput("G:\\refined.csv")
print("Ratio: ", DataC.getRatio())

#Apro il dataset
df = pd.read_csv("G:\\refined.csv")

#b = df.loc[df["src_ip"].str.contains("192.168.120")]