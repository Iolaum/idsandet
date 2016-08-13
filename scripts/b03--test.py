#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load validation data and count system call frequencies

from __future__ import division
import pickle
import numpy as np


print("Started processing validation data!")
#tmatrix = np.load("../data/b3_vamatrix.npy")

#tmatrix = np.load('../data/b1_trmatrix.npy')
tmatrix = np.load('../data/b2_atmatrix.npy')

print("tmatrix shape is {}".format(tmatrix.shape))

tsum = np.sum(tmatrix, axis=1)
print tsum
print np.sum(tsum)
