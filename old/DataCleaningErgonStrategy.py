# -*- coding: utf-8 -*-
"""
Alessandro Mini
This script will perform a deep-cleaning of the source dataset,
removing "noise" protocols and refining traffic according to
some policies.
A company or a firewall (should) have some specific rules to 
apply, such as time-delta and protocols to exclude.
"""

import pandas as pd 
from pathlib import Path
from DataCleanerInterface import DataCleaner

class ErgonDataCleaner(DataCleaner):

    def cleanDataset(self):
        print("gigi")

    def getRatio(self):
        print("gigi")
    def getBenchmark(self):
        print("gigi") 

verbose = True
path = Path("G:/30d_traffic_ergon.csv")
def vprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)
        
    
#List of protocols to exclude 
application_exclusions = ["ms-product-activation",
            "ms-update",
            "incomplete",
            "ms-ds-smbv3",
            "ntp-base",
            "icmp",
            "zabbix",
            "informix",
            "insufficient-data",
            "dhcp",
            "ms-local-security-management",
            "collectd",
            "ms-update-optimization-p2p",
            "ntp",
            "ping"]

vprint("processing ", path.name)
df = pd.read_csv(path)
len_before = len(df)
vprint("Removing junk data ...")
df = df[~df.src_port.eq(0)]
df = df[~df.dst_port.eq(0)]
df = df[~df.bytes_sent.eq(0)]
df = df[~df.bytes_recieved.eq(0)]

vprint("Applying protocols exclusion rules ...")
df = df[~df['application'].isin(application_exclusions)]




vprint("Removing night traffic from 18:00 to 8:00")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index('timestamp')
night = df.between_time('18:00', '8:00')
df = df.drop(night.index)
df = df.reset_index()





k = 2
vprint("Identifying sources with less than " , k , " hours of traffic ")
df["timestamp"] = pd.to_datetime(df["timestamp"])
grouped = df.groupby(pd.Grouper(freq="1H", key="timestamp"))
i = 0
sufficient = []
insufficient = []
for ip in df["src_ip"].unique():
    #print ("working ", ip ," ",i, " di ", len(df["src_ip"].unique()))
    found = 0
    i = i + 1 
    for hour in grouped.groups.keys():            
        try:
            a = grouped.get_group(hour)
            a = a.loc[a["src_ip"]==ip]
        except:
            found = 0 
        else:
            if(len(a)>1):      
                found = found + 1                
    if found >= k:
        sufficient.append(ip)              
    else:
        insufficient.append(ip)
    found = 0                          
      
    

df = df[~df['src_ip'].isin(insufficient)]
df = df.sort_values(by='timestamp', ascending=True)
len_after = len(df)
vprint("Dataset shrinked from ",len_before," to ", len_after, " ratio: " , round(len_before/len_after,2)*100, "%")

for label in sufficient:
    print(label, "  ", len(df.loc[df["src_ip"]==label]))
    
target = Path(path.name + "_refined")
df.to_csv("dataset_refined.csv",index= False)
    
    
    
    
    
    