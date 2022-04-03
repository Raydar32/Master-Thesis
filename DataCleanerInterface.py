# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:54:36 2022

@author: Alessandro
"""

from abc import ABC,abstractmethod
import pandas as pd 
from pathlib import Path
class DataCleaner(ABC):
    def vprint(self,*args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
            
    def setVerbose(self,verbose):
        self.verbose = verbose
        
    def loadDataset(self,datasetPath):
        self.datasetPath = Path(datasetPath)
        self.df = pd.read_csv(datasetPath)
        self.before_len = len(self.df)
        return self.df
        
    def setOutput(self,output):
        self.outputPath = Path(output)
        self.df.to_csv(self.outputPath,index=False)
        
    @abstractmethod
    def cleanDataset(self):
        pass
    

    def getRatio(self):
        return round(self.before_len/len(self.df),2)
    
    