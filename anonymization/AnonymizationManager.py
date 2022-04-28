# -*- coding: utf-8 -*-
"""
Script che genera la LUT e cifra/decifra un IP o un dataset

"""

from yacryptopan import CryptoPAn
from pathlib import Path
import os
import pandas as pd
import keyring
import sys

# keyring.set_password("up-kmae", "up-kmae", "8yo8A5XMbzhRe3mQbdymPnGdOOZMPuse"


LUTFileName = "LUT.csv"
LUTpath = Path(os.getcwd() + "/" + LUTFileName)
key = keyring.get_password("up-kmae", "up-kmae")
if key == None:
    print("System key for up-kmae has not been installed.\nFollow the documentation.")
    sys.exit(0)


def generateNewLUT():
    LUT = pd.DataFrame(columns=["original", "encrypted"])
    LUT.to_csv(LUTpath, index=False)
    LUT.set_index("original")


def anonymize_host(key, host):
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
    LUT = pd.read_csv(LUTpath)
    return LUT.loc[LUT.encrypted == host]["original"].to_string(index=False)


def anonymizeDataset(dataset):
    encrypted_dataset_index = dataset.index.to_series().apply(
        lambda t: anonymize_host(key, t))
    dataset.index = encrypted_dataset_index
    return dataset


def deanonymizeDataset(dataset):
    # Decrypt dataset
    decrypted_dataset_index = dataset.index.to_series().apply(
        lambda t: deanonymize_host(t))
    dataset.index = decrypted_dataset_index
    return dataset
