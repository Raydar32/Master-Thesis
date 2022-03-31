# -*- coding: utf-8 -*-
"""
Test file to keep everything together
"""



from ErgonDatasetCleaner import ErgonDataCleaner
DataC = ErgonDataCleaner()
DataC.setVerbose(True)
DataC.loadDataset("G:\\basic_dataset.csv")
DataC.cleanDataset()
DataC.setOutput("G:\\refined.csv")
print("Ratio: ", DataC.getRatio())

