# -*- coding: utf-8 -*-
"""
This script represents an abstract Interface for the data-cleaning
puropouse of UP-KMAE, each datacleaner (Refer to thesis.pdf) will
implement this interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class DataCleaner(ABC):

    def vprint(self, *args, **kwargs):
        """
        Method for printing verbosely or not.
        """
        if self.verbose:
            print(*args, **kwargs)

    def setVerbose(self, verbose):
        self.verbose = verbose

    def loadDataset(self, datasetPath):
        """
        Method to load a dataset (.csv extracted by datadump script) 

        """
        self.datasetPath = Path(datasetPath)
        self.df = pd.read_csv(datasetPath)
        self.before_len = len(self.df)
        return self.df

    def setOutput(self, output):
        """
        This will set the output of the dataCleaning

        """
        self.outputPath = Path(output)
        self.df.to_csv(self.outputPath, index=False)

    @abstractmethod
    def cleanDataset(self):
        pass

    def getRatio(self):
        return round(self.before_len/len(self.df), 2)
