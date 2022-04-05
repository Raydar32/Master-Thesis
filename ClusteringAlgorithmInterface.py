# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:44:25 2022

@author: Alessandro Mini
"""

from abc import ABC,abstractmethod
import pandas as pd 
from pathlib import Path

class ClusteringAlgorithm(ABC):
    def vprint(self,*args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
            
    def setVerbose(self,verbose):
        self.verbose = verbose
        
    #Setting a dataset
    def setData(self,df):
        self.df = df
        
    #Setoutput
    @abstractmethod
    def clusterize():
        pass