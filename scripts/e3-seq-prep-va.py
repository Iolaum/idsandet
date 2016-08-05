#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Preparing a short sequence feature engineering.

from __future__ import division
import pickle
import numpy as np

print("Creating 2-sequence system calls data object.")

# load 2sequence system calls dictionary
# dictionary setup !
#
# key-value-1
# syscal sequence : integer pointer
# (33, 79): 12
#
# key-value-2
# integer pointer : syscall sequence
# 12: (33, 79)
with open('../data/e1-sys-seq-dict.p', 'rb') as ha:
    sysdict = pickle.load(ha)


# load sequencis of training data
with open('../data/1_validation.p', 'rb') as ha:
    trdata = pickle.load(ha)


# create data object
trdata2 = np.zeros((len(trdata), 30626))

print("Populating data object.")

# dummy counter for data points
ctr = 0

# populate ...
for key, value in trdata.iteritems():
    # print key, value
    for it1 in range(len(value)-1):
        # print((value[it1], value[it1 + 1]))
        dummy1 = (value[it1], value[it1 + 1])
        trdata2[ctr, sysdict[dummy1]] += (1/(len(value)-1))
        #print trdata2[ctr, sysdict[dummy1]]
        
    if np.sum(trdata2[ctr, :]) != 1.0:
        print("Data point {} sum is {}.".format(ctr+1, np.sum(trdata2[ctr, :])))
    fraction = int(len(trdata)/100)
    if ctr % fraction == 0: 
        print("Population progress {}\%".format(int(ctr/fraction)))
    #exit()
    ctr += 1

datasums =  np.sum(trdata2, axis=0)
print("There are {} non zero 2-sequencies.".format(np.count_nonzero(datasums)))

with open('../data/e3-vadata2.p', 'wb') as ha:
    pickle.dump(trdata2, ha)

with open('../data/e3-va-datasums.p', 'wb') as ha:
    pickle.dump(datasums, ha)

print("Total probabilities: {}".format(np.sum(datasums)))
