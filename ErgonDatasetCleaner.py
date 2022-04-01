# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:06:29 2022

@author: Alessandro
"""

import pandas as pd 
from DataCleanerInterface import DataCleaner
from datetime import timedelta 
class ErgonDataCleaner(DataCleaner):                
    def cleanDataset(self):
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
    
        self.vprint("processing ", self.datasetPath.name)    
     
        self.vprint("Removing junk data ...")
        self.df =  self.df[~ self.df.src_port.eq(0)]
        self.df =  self.df[~ self.df.dst_port.eq(0)]
        self.df =  self.df[~ self.df.bytes_sent.eq(0)]
        self.df =  self.df[~ self.df.bytes_recieved.eq(0)]
    
        self.vprint("Applying protocols exclusion rules ...")
        self.df = self.df[~self.df['application'].isin(application_exclusions)]
    
        self.vprint("Removing night traffic from 18:00 to 8:00")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"],utc=True)
        self.df = self.df.set_index('timestamp')
        night = self.df.between_time('18:00', '8:00')
        self.df = self.df[~self.df.index.isin(night.index)]
        self.df = self.df.reset_index()        
        k = 3
        
        #Qui ci sono problemi, tronca il mio ip 
        self.vprint("Identifying sources with more then " , k , " hours of traffic ")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"],utc=True)
        
        #q = self.df.loc[self.df["src_ip"]=="192.168.121.47"]["timestamp"].iat[-1] - self.df.loc[self.df["src_ip"]=="192.168.121.47"]["timestamp"].iat[0] 
        
        grouped = self.df.groupby(pd.Grouper(freq="1H", key="timestamp"))
        i = 0
        sufficient = []
        insufficient = []
        for ip in self.df["src_ip"].unique():
           
            delta = self.df.loc[self.df["src_ip"]==ip]["timestamp"].iat[-1] - self.df.loc[self.df["src_ip"]==ip]["timestamp"].iat[0] 
            
            dbgflag = False
            if delta > timedelta(hours=k):
                sufficient.append(ip)
                dbgflag = True
            else:
                insufficient.append(ip)
            print(ip, " ",dbgflag)
                
    
                
        self.df = self.df[~self.df['src_ip'].isin(insufficient)]
        self.df = self.df.sort_values(by='timestamp', ascending=True)
        


            
            
            
            




