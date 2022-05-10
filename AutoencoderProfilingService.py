# -*- coding: utf-8 -*-
"""
This script implements the  service part of the REST Architecture, the flow of 
data is the follwing:
    
    [User] -> [API Rest] -> [ClusteringService]

So this script will implement the service part of the architecture, all the 
methods implemented here will be called by the REST API and will call methods
from the clustering interface.

"""

from models.AutoEncoderEmbeddingClustering import AutoencoderEmbeddingClusteringModel
from dataprocessing.FeatureExtractor import PaloAltoFeatureExtractor
from dataprocessing.DatasetCleaner import ErgonDataCleaner
import pandas as pd
import os.path
from pathlib import Path
from exceptions.modelNotFitException import modelNotFitException
from exceptions.itemNotFoundException import itemNotFoundException


def generateAssociationSet(targets, labelset):
    """ This method implements the creation and manteinance of an association set ip - cluster"""
    temp = labelset.loc[labelset.index.isin(
        targets)]

    table = pd.crosstab(index=temp.index,
                        columns=temp["cluster"], margins=False)
    return table


def adjustDatasetLength(df, days):
    """
    This method will resize the dataset to the previous x days.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mask = (df['timestamp'] >= df["timestamp"].max() - pd.Timedelta(days=days)) & (df['timestamp']
                                                                                   <= df["timestamp"].max())
    df = df.loc[mask]
    return df


class AutoencoderProfilingService():
    """
    This is the main class of the Autoencoder Profiling service, it exposes
    methods that will be called by the REST interface.
    This is set to work with the best performing methods.
    """

    def __init__(self):
        self.clusteringAlgorithm = AutoencoderEmbeddingClusteringModel(
            "isomap", "kmeans")
        self.score = None
        self.cluster_num = None
        self.association_set = None

    def getScore(self):
        if self.score == None:
            raise modelNotFitException("Model has not fit yet")
        else:
            return self.score

    def getClusterNum(self):
        if self.cluster_num == None:
            raise modelNotFitException("Model has not fit yet")
        else:
            return self.cluster_num

    def getAssociationSet(self):
        return self.association_set

    def getUserProfile(self, src_ip):
        try:
            association_set = self.getAssociationSet()
            found = association_set.loc[association_set.index == src_ip]
        except:
            raise modelNotFitException("Model has not fit yet")
        else:
            if len(found) == 0:
                raise itemNotFoundException("Item requested not found")
            else:
                return found

    def predictProfiles(self):
        """
        This method will perform the user profiling.
        """
        # Loading and cleaning the dataset
        print("Loading datasets")
        inputDataset = Path(os.getcwd() + "/datasets" + "/traffic_dataset.csv")
        outputCleanedDataset = Path(
            os.getcwd() + "/datasets" + "/" + "traffic_dataset_autoencoder" + "_r.csv")
        outputAssociationSet = Path(
            os.getcwd() + "/" + "autoencoder_association.csv")

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
