#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Merge Data and Split them

import numpy as np
from sklearn.cross_validation import train_test_split


trdata = np.load('../data/e6a-trdata.npy')
vadata = np.load('../data/e6b-vadata.npy')
atdata = np.load('../data/e6c-atdata.npy')

tvdata = np.concatenate((trdata, vadata), axis=0)
# delete unneeded data
del trdata
del vadata

# create classification labels
l1 = np.zeros(tvdata.shape[0])
l2 = np.ones(atdata.shape[0])
# print l2.shape


# split train set
xtr, xts, ytr, yts = train_test_split(np.concatenate((tvdata, atdata), axis=0), np.concatenate((l1,l2), axis=0), 
    test_size=0.3, random_state=69)


# delete unneeded data
del tvdata
del atdata
del l1
del l2

# save data
np.save('../data/e7a-xtr', xtr)
np.save('../data/e7b-ytr', ytr)
np.save('../data/e7c-xts', xts)
np.save('../data/e7d-yts', yts)


