# -*- coding: utf-8 -*-
"""
Spyder Editor

Service for clustering 
"""
from models.AutoEncoderEmbeddingClustering import AutoencoderEmbeddingClusteringModel
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner
import pandas as pd
import os.path
from pathlib import Path


def generateAssociationSet(targets, labelset):
    temp = labelset.loc[labelset.index.isin(
        targets)]

    table = pd.crosstab(index=temp.index,
                        columns=temp["cluster"], margins=False)
    return table


def adjustDatasetLength(df, days):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mask = (df['timestamp'] >= df["timestamp"].max() - pd.Timedelta(days=15)) & (df['timestamp']
                                                                                 <= df["timestamp"].max())
    df = df.loc[mask]
    return df


class AutoencoderProfilingService():
    def __init__(self):
        self.clusteringAlgorithm = AutoencoderEmbeddingClusteringModel(
            "isomap", "kmeans")
        self.score = None
        self.cluster_num = None
        self.association_set = None

    def getScore(self):
        if self.score == None:
            raise ValueError("Model has not fit yet")
        else:
            return self.score

    def getClusterNum(self):
        if self.cluster_num == None:
            raise ValueError("Model has not fit yet")
        else:
            return self.cluster_num

    def getAssociationSet(self):
        return self.association_set

    def getUserProfile(self, src_ip):
        association_set = self.getAssociationSet()
        return association_set.loc[association_set.index == src_ip]

    def predictProfiles(self):
        # Loading and cleaning the dataset
        inputDataset = Path(os.getcwd() + "/datasets" + "/traffic_dataset.csv")
        outputCleanedDataset = Path(
            os.getcwd() + "/datasets" + "/" + "traffic_dataset_autoencoder" + "_r.csv")
        outputAssociationSet = Path(
            os.getcwd() + "/" + "autoencoder_association.csv")

        if not os.path.isfile(inputDataset):
            raise Exception(
                "Input dataset in /datasets not found, no traffic_dataset.csv")

        # Cleaning Dataset
        DataC = ErgonDataCleaner()
        DataC.setMinimumHoursToBeClustered(10)
        DataC.setVerbose(True)
        DataC.loadDataset(inputDataset)
        DataC.cleanDataset()
        DataC.setOutput(outputCleanedDataset)
        print("Ratio: ", DataC.getRatio())
        df = pd.read_csv(outputCleanedDataset)

        # Adjusting lenght of dataset as experiments suggests:
        df = adjustDatasetLength(df, 15)

        # Extracting features
        featureExtractor = PaloAltoFeatureExtractor()
        featureExtractor.setEclusionList(None)
        extracted = featureExtractor.createAggregatedFeatureSet(df, "900s")

        # Starting clustering algorithm
        self.clusteringAlgorithm.setVerbose(True)
        self.clusteringAlgorithm.setData(extracted)
        labelset_kmeans_classic = self.clusteringAlgorithm.clusterize()
        print("Classic clustering beans (no-opt) ",
              self.clusteringAlgorithm.get_c_num(), " ", self.clusteringAlgorithm.get_score(), " ", )
        self.clusteringAlgorithm.show_plot(
            "Classical K-Means clustering (no-opt)")
        self.score = self.clusteringAlgorithm.get_score()
        self.cluster_num = self.clusteringAlgorithm.get_c_num()

        # Generating association set
        KMeansAssociationSet = generateAssociationSet(
            labelset_kmeans_classic.index.unique(), labelset_kmeans_classic)

        self.association_set = KMeansAssociationSet

        # Saving association set
        if os.path.isfile(outputAssociationSet):
            os.remove(outputAssociationSet)

        KMeansAssociationSet.to_csv(outputAssociationSet)

        if os.path.isfile(outputAssociationSet):
            return True
        else:
            return False


AutoencoderProfilingService = AutoencoderProfilingService()
AutoencoderProfilingService.predictProfiles()
AutoencoderProfilingService.getAssociationSet()
