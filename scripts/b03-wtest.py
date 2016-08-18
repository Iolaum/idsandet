#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load validation data and count system call frequencies

from __future__ import division
import pickle
import numpy as np


print("Started processing data!")


tmatrix = np.load('../data/b1_trmatrix.npy')

print("trmatrix shape is {}".format(tmatrix.shape))

tsum = np.sum(tmatrix, axis=1)
print np.sum(tsum)

tmatrix = np.load('../data/b2_atmatrix.npy')

print("atmatrix shape is {}".format(tmatrix.shape))

tsum = np.sum(tmatrix, axis=1)
print np.sum(tsum)


tmatrix = np.load("../data/b3_vamatrix.npy")

print("vamatrix shape is {}".format(tmatrix.shape))

tsum = np.sum(tmatrix, axis=1)
print np.sum(tsum)
