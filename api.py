# -*- coding: utf-8 -*-
"""
This script is the main interface for the rest API, it will expose the HTTP
routes and various methods to interact with.
The back-end is made up with Flask, in this basic version A.A.A systems
are not implemented but Auth0 could be a valid choice.
The service will be spawned by default in port 1000 but this can be changed.
Custom exceptions are implemented.

"""

from flask import Flask
from AutoencoderProfilingService import AutoencoderProfilingService
from KMeansProfilingService import KMeansProfilingService
import time
from exceptions.modelNotFitException import modelNotFitException
from exceptions.itemNotFoundException import itemNotFoundException

# Init Flask Application
app = Flask(__name__)

# Init profiler models
KMeansService = KMeansProfilingService()
AutoencoderService = AutoencoderProfilingService()


@app.route('/kmeans/<srcip>')
def getKmeansUserProfile(srcip):
    """
    This methods implements an HTTP get method that will return a profiler
    for a given ip.address es. http://localhost//kmeans//10.10.10.10, profiled
    using K-Means + f.s method (refer to thesis.pdf).
    """

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
    """
    This methods returns the score (Sihlouette score) associated with the
    k-means clustering technique.
    """
    score = 0
    try:
        score = str(KMeansService.getScore())
    except modelNotFitException:
        return "not fit yet", 404
    else:
        return score


@app.route('/aenc/<srcip>')
def getAencUserProfile(srcip):
    """
    This methods implements an HTTP get method that will return a profiler
    for a given ip.address es. http://localhost//kmeans//10.10.10.10, profiled
    using Autoencoder-Embedding method (refer to thesis.pdf).
    """
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
    """
    This methods returns the score (Sihlouette score) associated with Autoencoder
    Embedding clustering technique.
    """
    score = 0
    try:
        score = str(AutoencoderService.getScore())
    except modelNotFitException:
        return "not fit yet", 404
    else:
        return score


if __name__ == "__main__":
    """
    Main method, here the API will be started and model will be pre-fitted.
    """
    print("Fitting Kmeans and autoencoder model")
    app.logger.info('Fitting Kmeans and autoencoder model')
    start_time = time.time()

    AutoencoderService.predictProfiles()
    KMeansService.predictProfiles()

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Models fitted")

    # Api interface, here ported and IP can be changed,
    # default is 0.0.0.0:1000
    app.logger.info("Models fitted")
    app.run(debug=False, host='0.0.0.0', port=1000)
