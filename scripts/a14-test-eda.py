#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Get number of all system calls

from __future__ import division
import pickle
import numpy as np


# load sequencis of training data
with open('../data/1_training.p', 'rb') as ha:
    trdata = pickle.load(ha)

# system calls set
syset = set()

print("Populating data object.")

# populate ...
for key, value in trdata.iteritems():

    for it1 in range(len(value)):
        syset.add(value[it1])
    #exit()

# load sequencis of validation data
with open('../data/1_validation.p', 'rb') as ha:
    trdata = pickle.load(ha)

for key, value in trdata.iteritems():

    for it1 in range(len(value)):
        syset.add(value[it1])

# load sequencis of attack data
with open('../data/1_attack.p', 'rb') as ha:
    trdata = pickle.load(ha)

# populate ...
for key, value in trdata.iteritems():

    for it1 in range(len(value)):
        syset.add(value[it1])
    #exit()
print("There are {} system calls.".format(len(syset)))

with open('../data/a14-sys-set.p', 'wb') as ha:
    pickle.dump(syset, ha)


