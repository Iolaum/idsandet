#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement a kMC classifier
# set hardcoded variable to select attack methodology and distance

from __future__ import division
import pickle
import numpy as np
from os.path import exists
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


# custom print function to also save runs on text file
resultsfile = '../data/b7-kmc-1-results.txt'

# custom print function
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

myprint("Starting a k means clusting euclidean distance classifier.")

myprint("Loading trainind data.")
trdat = np.load('../data/b5_trmatrix.npy')

# use model if it exits
trainedmodel = '../data/b07-kmc-model-1.p'
if exists(trainedmodel):
    myprint('Loading trained kmc model.')
    with open(trainedmodel, 'rb') as ha:
        kmc = pickle.load(ha)
else:
    kmc = KMeans(n_clusters=5, n_jobs=-2, random_state=13)
    myprint('Training a k-means clustering model.')
    kmc.fit(trdat)
    myprint('Saving our kmc model.')
    with open(trainedmodel, 'wb') as ha:
        pickle.dump(kmc, ha)

# attack data options
atdatselector = 6

if atdatselector == 1:
    myprint("Loading adduser attack data.")
    atdat = np.load('../data/b5_at1mat_adduser.npy')

elif atdatselector == 2:
    myprint("Loading hydra ftp attack data.")
    atdat = np.load('../data/b5_at2mat_hyftp.npy')

elif atdatselector == 3:
    myprint("Loading hydra ssh attack data.")
    atdat = np.load('../data/b5_at3mat_hyssh.npy')

elif atdatselector == 4:
    myprint("Loading java meterpreter attack data.")
    atdat = np.load('../data/b5_at4mat_javamet.npy')

elif atdatselector == 5:
    myprint("Loading meterpreter attack data.")
    atdat = np.load('../data/b5_at5mat_meter.npy')

elif atdatselector == 6:
    myprint("Loading web shell attack data.")
    atdat = np.load('../data/b5_at6mat_webshell.npy')


myprint("Loading validation data.")
vadat = np.load('../data/b5_vamatrix.npy')


# sk learn: DeprecationWarning:
# Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19.
# Reshape your data either using X.reshape(-1, 1) if your data has a single feature
# or X.reshape(1, -1) if it contains a single sample.


# find d to split distances!
dist = 0.755064191429 #set me

# intermediate variable
cdist = 0

# will run 'once' and will be hardcoded
kmcents = kmc.cluster_centers_
if dist == 0:
    for it1 in trdat:
        cdist = euclidean(kmcents[kmc.predict(it1.reshape(1, -1))], it1)
        if cdist > dist:
            dist = cdist

        # debug
        # print kmcents
        # print kmcents[kmc.predict(it1.reshape(1, -1))]
        # print dist
        # exit()

    print("Maximum distance is: {}".format(dist))


# intermediate variables
dumctr = 0
totacc = atdat.shape[0]
totval = vadat.shape[0]

# barrier distance for classification
barr1 = 1 * dist  # setme!


# determine attack detection accuracy
for it1 in atdat:

    if euclidean(kmcents[kmc.predict(it1.reshape(1, -1))], it1) > barr1:
        dumctr += 1


myprint("With {} Euclidean distance limit for 5 k-means clusters\
 we get:".format(barr1))
myprint("Attack detection accuracy: {}".format(dumctr/totacc))


# determine false positive rate
# only need to be done once.
# it's the same across attacks
if atdatselector == 1:

    # determine false positive rate
    dumctr = 0
    for it1 in vadat:

        if euclidean(kmcents[kmc.predict(it1.reshape(1, -1))], it1) > barr1:
            dumctr += 1

    myprint("False positive rate: {}".format(dumctr/totval))

else:

    myprint("Read false positive rate from previous runs.")

with open(resultsfile, 'a') as ha:
    ha.write('\n')
