# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:19:18 2022

@author: Alessandro Mini
"""
import pandas as pd
import numpy as np

def extract_features(dataset):
    #grouby del dataset iniziale
    df_grouped = dataset.groupby("src_ip")
    
    #creo un data-frame di features da estrarre
    extracted = pd.DataFrame()
    
    #metto come index gli IP da profilare.
    extracted["src_ip"] = dataset["src_ip"].unique().copy()
    extracted = extracted.set_index("src_ip")
    
    # ---- Features a livello 3 -----
    # |dst_ip| : num di indirizzi IP diversi.
    extracted = extracted.join(
                    df_grouped["dst_ip"].nunique()
                )
    # |port_dst|: num di porte di destinazione diverse.
    extracted = extracted.join(
                    df_grouped["dst_port"].nunique()
                )
    # |src|: num di porte sorgente diverse.
    extracted = extracted.join(
                    df_grouped["src_port"].nunique()
                )
    
    # |dst_port|/|dst_ips|
    extracted = extracted.join(
                    (df_grouped["dst_port"].nunique()/df_grouped["dst_ip"].count()).rename("dst_diversity")
                )
    
    # |dst_port|/|dst_ips|
    extracted = extracted.join(
                    (df_grouped["src_port"].nunique()/df_grouped["dst_ip"].count()).rename("src_diversity")
                )
    # ---- Features a livello 4 ----
    
    #|udp|/|tot|
    extracted = extracted.join(
                    (dataset[dataset["transport"]=="udp"].groupby("src_ip")["transport"].count()/df_grouped["transport"].count()).replace(np.nan,0).rename("udp_ratio")
        )
    
    #|tcp|/|tot|
    extracted = extracted.join(
                    (dataset[dataset["transport"]=="tcp"].groupby("src_ip")["transport"].count()/df_grouped["transport"].count()).replace(np.nan,0).rename("tcp_ratio")
        )

    #|http|/|tot|
    extracted = extracted.join(
                    (dataset[ (dataset["dst_port"]==80) | (dataset["dst_port"]==443)].groupby("src_ip")["dst_port"].count()/df_grouped["dst_port"].count()).replace(np.nan,0).rename("http_ratio")
        )
    
    #|ssh|/|tot|
    extracted = extracted.join(
                    (dataset[ (dataset["dst_port"]==22)].groupby("src_ip")["dst_port"].count()/df_grouped["dst_port"].count()).replace(np.nan,0).rename("ssh_ratio")
                    
        )

    #|smtp|/|tot|
    extracted = extracted.join(
                    (dataset[ (dataset["dst_port"]==25) | (dataset["dst_port"]==587)| (dataset["dst_port"]==465)].groupby("src_ip")["dst_port"].count()/df_grouped["dst_port"].count()).replace(np.nan,0).rename("smtp_ratio")
        )
        
    #|tot bytes|
    
    #|tot packets|
    
    #|avg bytes|
    
    #|avg packets|
    
    #|applicaitons|
    #extracted = extracted.join(
    #                df_grouped["application"].nunique()
    #           )
    

    #|duration|
    extracted = extracted.join(
                    df_grouped["duration"].mean().rename("avg_duration") 
                )
    
    return extracted


df = pd.read_csv("C:\\InProgress\\Tesi\\trafficDump\\1.04to20.02_1d_test.csv")
extracted = extract_features(df)

