# -*- coding: utf-8 -*-
"""
This script will perform the pseudoanonimyzation task, using 
CryptoPan implementation from Xu. Et. Al (2002). The main idea 
is to transform private data in order to be "presentable" for this 
thesis.
First thing to do is to set-up a system-level encryption password
for this application, this should be a 32 bit string.

keyring.set_password("up-kmae", "up-kmae", "32bitkey")

Then you can use this module, for now its usage shall be done manually
calling methods, in future it will be integrated at some point in UP-KMAE
structure.                     

"""

from yacryptopan import CryptoPAn
from pathlib import Path
import os
import pandas as pd
import keyring
import sys


# This file represent the LUT (lookup table) path in the system.
LUTFileName = "LUT.csv"
LUTpath = Path(os.getcwd() + "/" + LUTFileName)
# Here the script will access the password
key = keyring.get_password("up-kmae", "up-kmae")
if key == None:
    print("System key for up-kmae has not been installed.\nFollow the documentation.")
    sys.exit(0)


def generateNewLUT():
    """
    Method that creates a new pseudoanonymization LUT if it does not exists.
    """
    LUT = pd.DataFrame(columns=["original", "encrypted"])
    LUT.to_csv(LUTpath, index=False)
    LUT.set_index("original")


def anonymize_host(key, host):
    """
    Method that implements anonymization of a single host.
    """
    if not os.path.isfile(LUTpath):
        generateNewLUT()

    LUT = pd.read_csv(LUTpath)
    if len(LUT.loc[LUT.original == host]) != 0:
        return LUT.loc[LUT.original == host]["encrypted"].to_string(index=False)
    else:
        cp = CryptoPAn(str.encode(key))
        encrypted = cp.anonymize(host)
        LUT = LUT.append(
            {"original": host, "encrypted": encrypted}, ignore_index=True)
        LUT.to_csv(LUTpath, index=False)

    return LUT.loc[LUT.original == host]["encrypted"].to_string(index=False)


def deanonymize_host(host):
    """
    Method that implements de-anonymization of a single host.
    """
    LUT = pd.read_csv(LUTpath)
    return LUT.loc[LUT.encrypted == host]["original"].to_string(index=False)


def anonymizeDataset(dataset):
    """
    Method that encrypts a whole dataset-
    """
    encrypted_dataset_index = dataset.index.to_series().apply(
        lambda t: anonymize_host(key, t))
    dataset.index = encrypted_dataset_index
    return dataset


def deanonymizeDataset(dataset):
    """
    Method that decrypts a whole dataset-
    """
    decrypted_dataset_index = dataset.index.to_series().apply(
        lambda t: deanonymize_host(t))
    dataset.index = decrypted_dataset_index
    return dataset
