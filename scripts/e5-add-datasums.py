#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Merge Data and Split them

from __future__ import division
import pickle
import numpy as np


with open('../data/e2-tr-datasums.p', 'rb') as ha:
    trsums = pickle.load(ha)

print("{} non zero elements in trdata.".format(np.count_nonzero(trsums)))

with open('../data/e3-va-datasums.p', 'rb') as ha:
    atsums = pickle.load(ha)

print("{} non zero elements in atdata.".format(np.count_nonzero(atsums)))

with open('../data/e4-at-datasums.p', 'rb') as ha:
    vasums = pickle.load(ha)

print("{} non zero elements in vadata.".format(np.count_nonzero(vasums)))

totsums = np.add(trsums, atsums)
totsums = np.add(totsums, vasums)

print("Total data dimensions: {}".format(totsums.shape[0]))
print("{} non zero elements in total sum.".format(np.count_nonzero(totsums)))

# find zero 2seq calls
indx = []

for it in range(30626):
    if totsums[it]==0.:
        indx.append(it)

print("Found {} zero frequency entries".format(len(indx)))


# sodr in descending order so deleting is consistent
indx.sort(reverse=True)

# save indexes to be deleted!
with open('../data/e5-zeroindex.p', 'wb') as ha:
    pickle.dump(indx, ha)

# 2freq sys list
syscalls = np.array(range(30626))

# delete uncalled system calls
for it in indx:
    syscalls = np.delete(syscalls, it, 0)

# save list of kept 2seq sys calls
with open('../data/e5-2freqsysc.p', 'wb') as ha:
    pickle.dump(syscalls, ha)


