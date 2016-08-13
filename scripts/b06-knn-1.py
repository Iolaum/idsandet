#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement a kNN classifier
# set hardcoded variable to select attack methodology

from __future__ import division
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
import math


# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open('../data/b6-knn-1-results.txt', 'a') as ha:
        ha.write(mytext + '\n')

myprint("Starting a kNN squared euclidean distance classifier.")

myprint("Loading training data.")
trdat = np.load('../data/b5_trmatrix.npy')

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


# intermediate variables
ctracc = 0
ctrfpr = 0
dumctr = 0
totacc = atdat.shape[0]
totval = vadat.shape[0]

# euclidian distance radius
barr1 = 0.1

# normal data neighbours thresshold
barr2 = 20


# determine attack detection accuracy
for it1 in atdat:

    # debug
    # print type(it1)
    # print it1.shape
    # exit()

    for it2 in trdat:
        if math.pow(euclidean(it1, it2), 2) <= barr1:
            dumctr += 1

    if dumctr < barr2:
        ctracc += 1

    dumctr = 0


myprint("With {} squared Euclidean distance limit and {} neighbours limit we get:".format(barr1, barr2))
myprint("Attack detection accuracy: {}".format(ctracc/totacc))


# only need to be done once.
# it's the same across attacks
if atdatselector == 1:

    # determine false positive rate
    for it1 in vadat:

        # debug
        # print type(it1)
        # print it1.shape
        # exit()

        for it2 in trdat:
            if math.pow(euclidean(it1, it2), 2) <= barr1:
                dumctr += 1

        if dumctr < barr2:
            ctrfpr += 1

        dumctr = 0

    myprint("False Positive rate: {}".format(ctrfpr/totval))

else:

    myprint("Read false positive rate from previous runs.")


with open('../data/b6-knn-1-results.txt', 'a') as ha:
    ha.write('\n')
