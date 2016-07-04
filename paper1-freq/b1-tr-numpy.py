#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle
import numpy as np


print("Scipt starting!\nReading training data file!")


with open('../data/training.p', 'rb') as ha:
    trdata = pickle.load(ha)


# create numpy array to handle the data
# first number is lines, second is columns.
# 833 training data points
# 325 system calls to check their frequency.
tmatrix = np.zeros((833, 325))

# counter for line in array to be added
linectr = 0


syscalls = range(1, 326)

for key in trdata:
    dummy1 = np.asarray(trdata[key])
    len1 = len(dummy1)

    # sys call counter
    ctr = 0

    # count system call frequency and fill Tr data matrix
    for ic in syscalls:
        for it in trdata[key]:
            if it == ic:
                ctr += 1
        tmatrix[linectr, (ic-1)] = ctr/len1

        # reset sys call counter
        ctr = 0

    # debug
    #for it in tmatrix[0,:]:
    #    if it > 0:
    #        print it
    #break
    # debug

    # move to next row of tmatrix
    linectr += 1

    # debug
    #if linectr == 5:
    #    break


with open('../data/b1_tmatrix.p', 'wb') as ha:
    pickle.dump(tmatrix, ha)

print("tmatrix saved as nd array with pickle.")
