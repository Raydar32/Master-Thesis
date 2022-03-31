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

#List of protocols to exclude 
exclusion = ["ms-product-activation",
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


df = pd.read_csv("C:\\Users\\Alessandro Mini\\Downloads\\30d_traffic_ergon.csv")


#Applying exclusion rules
print("Applying protocols exclusion rules")
len_before = len(df)
df = df[~df['application'].isin(exclusion)]
len_after = len(df)
print("----------------------------------------------------")
print("Bef: " , len_before , " after: ", len_after)
print("Removed data : " ,len_before/len_after)
print("----------------------------------------------------")


#In questo punto bisogna inserire la rimozione dei record
#in orario notturno.

#df["timestamp"].between_time('18:00', '8:00')
print("Removing night traffic from 18:00 to 8:00")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index('timestamp')
night = df.between_time('18:00', '8:00')
df = df.drop(night.index,inplace = True)
df.reset_index()





k = 10
print("Removing sources with less than " , k , " hours of traffic ")
#Find sources with less than k hours of traffic

df["timestamp"] = pd.to_datetime(df["timestamp"])
grouped = df.groupby(pd.Grouper(freq="1H", key="timestamp"))
#keys_uniques = grouped.groups.keys()
#ora = grouped.get_group("2022-03-18 17:00:00+0100")
#Per tutti gli indirizzi ip, scorri tutte le ore
#Se un indirizzo compare  in > 3 risparmialo e salvalo, altrimenti 
#eliminalo (in questa prima fase, perchè poi magari si butta in un
#cluster, oppure si costruisce una tabella)


i = 0
sufficient = []
insufficient = []
for ip in df["src_ip"].unique():
    print ("working ", ip ," ",i, " di ", len(df["src_ip"].unique()))
    found = 0
    i = i + 1 
    for hour in grouped.groups.keys():        
    
        a = grouped.get_group(hour)
        a = a.loc[a["src_ip"]==ip]
        if(len(a)>1):      
            found = found + 1                
    if found >= k:
        sufficient.append(ip)              
    else:
        insufficient.append(ip)
    found = 0                          
      
print("Removing IPs with less then ", k , " hours of traffic ")
df = df[~df['src_ip'].isin(insufficient)]
len_after = len(df)
print("----------------------------------------------------")
print("Bef: " , len_before , " after: ", len_after)
print("Removed data : " ,len_before/len_after)
print("----------------------------------------------------")




#Creare un nuovo CSV: Dataset_refined.csv
            

#Print della quantità di record per IP univoco
#for label in df["src_ip"].unique():
#    print(label, "  ", len(df.loc[df["src_ip"]==label]))
    
    
    
    
    
    
    
    
    