# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""



from ErgonDatasetCleaner import ErgonDataCleaner
import pandas as pd 
#------------------- Pulisco il dataset -------------------
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("C:\\InProgress\\Tesi\\20.02to30.03.csv")
DataC.cleanDataset()
DataC.setOutput("C:\\InProgress\\Tesi\\20.02to30.03_refined.csv")
print("Ratio: ", DataC.getRatio())

#Apro il dataset
df = pd.read_csv("C:\\InProgress\\Tesi\\20.02to30.03_refined.csv")

#b = df.loc[df["src_ip"].str.contains("192.168.120")]