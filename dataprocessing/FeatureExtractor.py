# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:19:18 2022

@author: Alessandro Mini
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MaxAbsScaler


class PaloAltoFeatureExtractor():

    def __init__(self):
        self.exclusionList = None

    def setEclusionList(self, exclusionList):
        self.exclusionList = exclusionList

    def createAggregatedFeatureSet(self, dataset, aggregationTime):
        extracted = pd.DataFrame()
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
        grouped = dataset.groupby(pd.Grouper(
            freq=aggregationTime, key="timestamp"))
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
                extracted = extracted.append(
                    self.extractFeaturesFromDataSlice(a))
            i = i + 1
            print("Scanning time-slot :", i, " di ", len(grouped.groups.keys()), hour.hour, ":", hour.minute, " giorno ", hour.day, "//",
                  hour.month, " totale ", len(grouped.groups.keys()), esito)

        extracted_values = MaxAbsScaler().fit_transform(extracted)
        extracted = pd.DataFrame(
            extracted_values, index=extracted.index, columns=extracted.columns)
        extracted = extracted[(
            np.abs(stats.zscore(extracted)) < 3).all(axis=1)]

        extracted["ssh_ratio"] = extracted["ssh_ratio"].apply(
            lambda x: 1 if x > 0 else 0)

        # Applico la lsita di esclusioni
        if self.exclusionList != None:
            for item in self.exclusionList:
                print("Rimuovo feature : ", item)
                del extracted[item]
        return extracted

    def extractFeaturesFromDataSlice(self, dataSlice):
        # grouby del dataSlice iniziale
        dataSliceGroupedBySourceIP = dataSlice.groupby("src_ip")

        # creo un data-frame di features da estrarre
        extractedFeatures = pd.DataFrame()

        # metto come index gli IP da profilare.
        extractedFeatures["src_ip"] = dataSlice["src_ip"].unique().copy()
        extractedFeatures = extractedFeatures.set_index("src_ip")

        # ---- Features a livello 3 -----
        # |dst_ip| : num di indirizzi IP diversi.
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["dst_ip"].nunique()
        )
        # |port_dst|: num di porte di destinazione diverse.
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["dst_port"].nunique()
        )
        # |src|: num di porte sorgente diverse.
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["src_port"].nunique()
        )

        # |dst_port|/|dst_ips|
        extractedFeatures = extractedFeatures.join(
            (dataSliceGroupedBySourceIP["dst_port"].nunique(
            )/dataSliceGroupedBySourceIP["dst_ip"].count()).rename("dst_diversity")
        )

        # |dst_port|/|dst_ips|
        extractedFeatures = extractedFeatures.join(
            (dataSliceGroupedBySourceIP["src_port"].nunique(
            )/dataSliceGroupedBySourceIP["dst_ip"].count()).rename("src_diversity")
        )
        # ---- Features a livello 4 ----

        # |udp|/|tot|
        extractedFeatures = extractedFeatures.join(
            (dataSlice[dataSlice["transport"] == "udp"].groupby("src_ip")["transport"].count(
            )/dataSliceGroupedBySourceIP["transport"].count()).replace(np.nan, 0).rename("udp_ratio")
        )

        # |tcp|/|tot|
        extractedFeatures = extractedFeatures.join(
            (dataSlice[dataSlice["transport"] == "tcp"].groupby("src_ip")["transport"].count(
            )/dataSliceGroupedBySourceIP["transport"].count()).replace(np.nan, 0).rename("tcp_ratio")
        )

        # |http|/|tot| (Include protocollo QUIC)
        extractedFeatures = extractedFeatures.join(
            (dataSlice[(dataSlice["dst_port"] == 80) | (dataSlice["dst_port"] == 443)].groupby("src_ip")[
             "dst_port"].count()/dataSliceGroupedBySourceIP["dst_port"].count()).replace(np.nan, 0).rename("http_ratio")
        )

        # |ssh|/|tot| (Normalizza a 1, feature del tipo isON)
        extractedFeatures = extractedFeatures.join(
            (dataSlice[(dataSlice["dst_port"] == 22)].groupby("src_ip")["dst_port"].count(
            )/dataSliceGroupedBySourceIP["dst_port"].count()).replace(np.nan, 0).rename("ssh_ratio")

        )

        # |smtp|/|tot|
        extractedFeatures = extractedFeatures.join(
            (dataSlice[(dataSlice["dst_port"] == 25) | (dataSlice["dst_port"] == 587) | (dataSlice["dst_port"] == 465) & (dataSlice["transport"] == "tcp")].groupby(
                "src_ip")["dst_port"].count()/dataSliceGroupedBySourceIP["dst_port"].count()).replace(np.nan, 0).rename("smtp_ratio")
        )

        # |tot bytes|
        extractedFeatures = extractedFeatures.join(
            (dataSliceGroupedBySourceIP["bytes_src"].sum(
            ) + dataSliceGroupedBySourceIP["bytes_dst"].sum()).rename("bytes")
        )

        # |tot packets|
        extractedFeatures = extractedFeatures.join(
            (dataSliceGroupedBySourceIP["packets_src"].sum(
            ) + dataSliceGroupedBySourceIP["packets_dst"].sum()).rename("packets")
        )

        # |bytes src avg|
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["bytes_src"].mean().rename(
                "src_avg_bytes")
        )

        # |bytes dst avg|
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["bytes_dst"].mean().rename(
                "dst_avg_bytes")
        )

        # |packets src avg|
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["packets_src"].mean().rename(
                "packets_src_avg")
        )

        # |packets dst avg|
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["packets_src"].mean().rename(
                "packets_dst_avg")
        )

        # |Avg duration|
        extractedFeatures = extractedFeatures.join(
            (dataSliceGroupedBySourceIP["duration"].mean(
            )/1000000000).rename("avg_duration")
        )

        return extractedFeatures
