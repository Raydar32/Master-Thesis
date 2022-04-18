# -*- coding: utf-8 -*-
"""
Flask REST interface
@author: Alessandro
"""

from flask import Flask
from AutoencoderProfilingService import AutoencoderProfilingService
from KMeansProfilingService import KMeansProfilingService
import time

# Init Flask Application
app = Flask(__name__)

# Profilers
KMeansService = KMeansProfilingService()
AutoencoderService = AutoencoderProfilingService()


@app.route('/kmeans/<srcip>')
def getKmeansUserProfile(srcip):
    return KMeansService.getUserProfile(srcip).to_dict()


@app.route('/kmeans/score')
def getKmeansScore():
    return str(KMeansService.getScore())


@app.route('/aenc/<srcip>')
def getAencUserProfile(srcip):
    return AutoencoderService.getUserProfile(srcip).to_dict()


@app.route('/aenc/score')
def getAencScore():
    return str(AutoencoderService.getScore())


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
