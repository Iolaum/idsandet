#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement a kNN classifier
# set hardcoded variable to select attack methodology

from __future__ import division
import pickle
import numpy as np
from scipy.spatial.distance import seuclidean
import math


# custom print function to also save runs on text file
resultsfile = '../data/b6-knn-2-results.txt'
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

myprint("Starting a kNN squared standardised euclidean distance classifier.")

myprint("Loading trainind data.")
with open('../data/b5_trmatrix.p', 'rb') as ha:
    trdat = pickle.load(ha)

# compute variance vector
datavar = np.var(trdat, axis=0)


# debug
# print datavar
# exit()

# attack data options
atdatselector = 1

if atdatselector == 1:
    myprint("Loading adduser attack data.")
    with open('../data/b5_at1mat_adduser.p', 'rb') as ha:
        atdat = pickle.load(ha)

elif atdatselector == 2:
    myprint("Loading hydra ftp attack data.")
    with open('../data/b5_at2mat_hyftp.p', 'rb') as ha:
        atdat = pickle.load(ha)

elif atdatselector == 3:
    myprint("Loading hydra ssh attack data.")
    with open('../data/b5_at3mat_hyssh.p', 'rb') as ha:
        atdat = pickle.load(ha)

elif atdatselector == 4:
    myprint("Loading java meterpreter attack data.")
    with open('../data/b5_at4mat_javamet.p', 'rb') as ha:
        atdat = pickle.load(ha)

elif atdatselector == 5:
    myprint("Loading meterpreter attack data.")
    with open('../data/b5_at5mat_meter.p', 'rb') as ha:
        atdat = pickle.load(ha)

elif atdatselector == 6:
    myprint("Loading web shell attack data.")
    with open('../data/b5_at6mat_webshell.p', 'rb') as ha:
        atdat = pickle.load(ha)


myprint("Loading validation data.")
with open('../data/b5_vamatrix.p', 'rb') as ha:
    vadat = pickle.load(ha)


# intermediate variables
ctracc = 0
ctrfpr = 0
dumctr = 0
totacc = atdat.shape[0]
totval = vadat.shape[0]

# metric distance radius
barr1 = 1

# normal data neighbours thresshold
barr2 = 20


# determine attack detection accuracy
for it1 in atdat:

    # debug
    # print type(it1)
    # print it1.shape
    # exit()

    for it2 in trdat:
        if math.pow(seuclidean(it1, it2, datavar), 2) <= barr1:
            dumctr += 1

    if dumctr < barr2:
        ctracc += 1

    dumctr = 0


myprint("With {} Standardised square Euclidean distance limit and {} neighbours\
 limit we get:".format(barr1, barr2))
myprint("Attack detection accuracy: {}".format(ctracc/totacc))


# determine false positive rate
for it1 in vadat:

    # debug
    # print type(it1)
    # print it1.shape
    # exit()

    for it2 in trdat:
        if math.pow(seuclidean(it1, it2, datavar), 2) <= barr1:
            dumctr += 1

    if dumctr < barr2:
        ctrfpr += 1

    dumctr = 0

myprint("False Positive rate: {}".format(ctrfpr/totval))

with open(resultsfile, 'a') as ha:
    ha.write('\n')
