#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# delete 0 features
import pickle
import numpy as np

# load indexes to be deleted!
with open('../data/e05-zeroindex.p', 'rb') as ha:
    indx = pickle.load(ha)

# load vadata
trdata = np.load('../data/e04-atdata2.npy')

# debug
# print trdata.shape
# print len(indx)
# print indx[2]
# exit()

tot1 = len(indx)
frct = tot1/100
#print frct
#exit()

ctr = 0

# delete uncalled system calls
for it in indx:
    trdata = np.delete(trdata, it, 1)

    if ctr%frct == 0:
        print("Progress at {}%".format(ctr/frct)) 

    ctr +=1

# verify
print("atdata shape is {}".format(trdata.shape))

# save data
np.save('../data/e06c-atdata', trdata)
