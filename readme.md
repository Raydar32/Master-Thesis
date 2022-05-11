# UP-KMAE

In this project for my Master's Thesis at University Of Florence (Italy) our aim is to build a system that can profile network users activities using firewall logs.

In this (experimental) version we use a (private) dataset from a local company that allowed our research as a traineship project.

# Findings
What we have found in this work, starting from previous litterature, is a new method to profile users in a network starting from their traffic, in this case we 
use logs from a Palo Alto PA-200 firewall appliance.
Sihlouette score is the main metric used to determine profiling (clustering) quality, we studied both classical and new "deep learning" models, implementing
both K-Means and Autoencoder-Embedding clustering such as R.MConville work N2D.
We achieved SC metrics around 0.93 for K-Means and 0.85 for Autoencoder (Fully unsupervised), these results are obtained with particular feature selection
techniques for the first one and dataset optimization for the second.
The system we got is completely modular thanks to OOP principles and implementable in a real corporate network, for further informations just refer to 
thesis.pdf that soon will be translated in english.

## Installation
To install the system you can do it in "classic" manner: 

```bash
pip install requirements.txt
```
Then place the dataset named "traffic_dataset".csv in the /datasets/ folder.

Or by using Docker, the second option for now is experimental but the project contains a Dockerfile so You can just build the image and then run it.


## Usage

To run the project:

```bash
python3 api.py
```
It will deploy a (local) REST API @ 0.0.0.0:1000, you can also run it by docker-run if you decided to use Docker.


## Results
Here some results will be added


## License
[MIT](https://choosealicense.com/licenses/mit/)