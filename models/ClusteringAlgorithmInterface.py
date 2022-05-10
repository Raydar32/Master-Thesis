# -*- coding: utf-8 -*-
"""
This script implements a generic interface for a clustering algorithm 
or methodology.
"""

from abc import ABC, abstractmethod


class ClusteringAlgorithm(ABC):
    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def setVerbose(self, verbose):
        self.verbose = verbose

    # Setting a dataset
    def setData(self, df):
        self.df = df

    # Setting output
    @abstractmethod
    def clusterize():
        pass
