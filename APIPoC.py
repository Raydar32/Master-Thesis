# -*- coding: utf-8 -*-
"""
Flask REST interface
@author: Alessandro
"""

from flask import Flask
from AutoencoderProfilingService import AutoencoderProfilingService
from KMeansProfilingService import KMeansProfilingService
import time
from flask import abort
from exceptions.modelNotFitException import modelNotFitException
from exceptions.itemNotFoundException import itemNotFoundException

# Init Flask Application
app = Flask(__name__)

# Profilers
KMeansService = KMeansProfilingService()
AutoencoderService = AutoencoderProfilingService()


@app.route('/kmeans/<srcip>')
def getKmeansUserProfile(srcip):
    found = None
    try:
        found = KMeansService.getUserProfile(srcip).to_dict()
    except modelNotFitException:
        return "not fit yet", 404
    except itemNotFoundException:
        return str(srcip + " not found "), 404
    else:
        return found


@app.route('/kmeans/score')
def getKmeansScore():
    score = 0
    try:
        score = str(KMeansService.getScore())
    except modelNotFitException:
        return "not fit yet", 404
    else:
        return score


@app.route('/aenc/<srcip>')
def getAencUserProfile(srcip):
    found = None
    try:
        found = AutoencoderService.getUserProfile(srcip).to_dict()
    except modelNotFitException:
        return "not fit yet", 404
    except itemNotFoundException:
        return str(srcip + " not found "), 404
    else:
        return found


@app.route('/aenc/score')
def getAencScore():
    score = 0
    try:
        score = str(AutoencoderService.getScore())
    except modelNotFitException:
        return "not fit yet", 404
    else:
        return score


@app.route('/ping')
def ping():
    return "pong"


if __name__ == "__main__":
    print("Fitting Kmeans and autoencoder model")
    start_time = time.time()

    KMeansService.predictProfiles()
    AutoencoderService.predictProfiles()

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Models fitted")

    app.run(host='0.0.0.0', port=1000, debug=False)
