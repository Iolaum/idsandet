#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load validation data and count system call frequencies

from __future__ import division
import pickle
import numpy as np


print("Started processing validation data!")

# laod validation data
with open('../data/1_validation.p', 'rb') as ha:
    trdata = pickle.load(ha)

# load non-zero sys calls
with open('../data/a14-sys-list.p', 'rb') as ha:
    syslist = pickle.load(ha)


# create numpy array to handle the data
# first number is lines, second is columns.
tmatrix = np.zeros((4372, 175))

# counter for line in array to be added
linectr = 0


for key in trdata:
    dummy1 = np.asarray(trdata[key])
    len1 = len(dummy1)

    # sys call counter
    ctr = 0

    # count system call frequency and fill Tr data matrix
    for ic in range(len(syslist)):
        for it in trdata[key]:
            if it == syslist[ic]:
                ctr += 1
        tmatrix[linectr, ic] = ctr/len1

        # reset sys call counter
        ctr = 0

np.save("../data/b3_vamatrix", tmatrix)

print("vamatrix saved as npy array with numpy.")
