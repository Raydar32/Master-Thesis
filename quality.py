# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 


exclusion = ["ms-product-activation",
            "ms-update",
            "incomplete",
            "ms-ds-smbv3",
            "ntp-base",
            "dns",
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

len_before = len(df)
df = df[~df['application'].isin(exclusion)]
len_after = len(df)
print("----------------------------------------------------")
print("Bef: " , len_before , " after: ", len_after)
print("Traffico inutile (rimosso) " ,len_after/len_before)
print("----------------------------------------------------")
for label in df["src_ip"].unique():
    print(label, "  ", len(df.loc[df["src_ip"]==label]))
    
    
    
    
    
    
    
    
    