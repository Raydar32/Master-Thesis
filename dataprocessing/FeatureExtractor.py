# -*- coding: utf-8 -*-
"""
This script here implements a procedure for extracting features
for Ergon s.r.l with Palo Alto firewall, it will do all the basics steps
of a data science/analysis project, such as:
    - Feature extraction
    - Normalization
    - Outlier reoval
"""
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import integrate, stats


def IsolationForestRemoveOutliers(dataset):
    """
    Method that implements IsolationForest, one of the techniques used for 
    outlier removal, the parameter c is determined with experiments.
    """
    outlierRemoval = IsolationForest(contamination=0.01)
    predictedOutliers = outlierRemoval.fit_predict(dataset)
    dataset = dataset[np.where(predictedOutliers == 1, True, False)]
    return dataset


class PaloAltoFeatureExtractor():

    def __init__(self):
        self.exclusionList = None

    def setEclusionList(self, exclusionList):
        """
            Setting up a protocol exclusion list, for information refer to 
            api.py
        """
        self.exclusionList = exclusionList

    def createAggregatedFeatureSet(self, dataset, aggregationTime):
        """
        This method will create an aggregated feature set that will be used
        by learning algorithms.
        """

        # Feature extraction phase
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

        # Normalization Phase
        extracted_values = MinMaxScaler().fit_transform(extracted)
        extracted = pd.DataFrame(
            extracted_values, index=extracted.index, columns=extracted.columns)

        # Outlier detection phase
        print("Outlier detection and removal")
        extracted = extracted[(
            np.abs(stats.zscore(extracted)) < 3).all(axis=1)]

        # Linearization of ratio in 0/1 mode.
        extracted["ssh_ratio"] = extracted["ssh_ratio"].apply(
            lambda x: 1 if x > 0 else 0)

        extracted["rdp_ratio"] = extracted["rdp_ratio"].apply(
            lambda x: 1 if x > 0 else 0)

        # Removing exclusion list protocols.
        if self.exclusionList != None:
            for item in self.exclusionList:
                del extracted[item]
        return extracted

    def extractFeaturesFromDataSlice(self, dataSlice):
        """
        This methods takes in a data slice of amplitude T and 
        extracts features, the following features are part of 
        the thesis as a research project.
        """
        # Groupby source IP of the original dataset.
        dataSliceGroupedBySourceIP = dataSlice.groupby("src_ip")

        # Creating a data-frame
        extractedFeatures = pd.DataFrame()

        # Setting up the src_ip as index.
        extractedFeatures["src_ip"] = dataSlice["src_ip"].unique().copy()
        extractedFeatures = extractedFeatures.set_index("src_ip")

        # Features will be both level 3 and level 4.

        # ---------------------------------------
        #            Level 3 Features
        # ---------------------------------------

        # |dst_ip| : Num. of different IP addresses for destinations.
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["dst_ip"].nunique()
        )

        # |port_dst|: Num. of unique dst.ports.
        extractedFeatures = extractedFeatures.join(
            dataSliceGroupedBySourceIP["dst_port"].nunique()
        )

        # |src|: Num. of unique src. ports.
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

        # ---------------------------------------
        #            Level 4 Features
        # ---------------------------------------

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

        # |http|/|tot| (QUIC)
        extractedFeatures = extractedFeatures.join(
            (dataSlice[(dataSlice["dst_port"] == 80) | (dataSlice["dst_port"] == 443)].groupby("src_ip")[
             "dst_port"].count()/dataSliceGroupedBySourceIP["dst_port"].count()).replace(np.nan, 0).rename("http_ratio")
        )

        # |ssh|/|tot|
        extractedFeatures = extractedFeatures.join(
            (dataSlice[(dataSlice["dst_port"] == 22)].groupby("src_ip")["dst_port"].count(
            )/dataSliceGroupedBySourceIP["dst_port"].count()).replace(np.nan, 0).rename("ssh_ratio")

        )

        # |rdp|/|tot|
        extractedFeatures = extractedFeatures.join(
            (dataSlice[(dataSlice["dst_port"] == 3389)].groupby("src_ip")["dst_port"].count(
            )/dataSliceGroupedBySourceIP["dst_port"].count()).replace(np.nan, 0).rename("rdp_ratio")

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
