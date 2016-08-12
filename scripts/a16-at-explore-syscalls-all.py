#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

import pickle
import numpy as np

print("Started processing attack data!")


with open('../data/1_attack.p', 'rb') as ha:
    trdata = pickle.load(ha)

# load non-zero sys calls
with open('../data/a14-sys-set.p', 'rb') as ha:
    syset = pickle.load(ha)

syslist = list(syset)
syslist.sort
del syset


# create numpy array to handle the data
tmatrix = np.zeros((1, len(syslist)))


for key in trdata:

    # sys call counter
    ctr = 0

    # count system call frequency and fill Tr data matrix
    for ic in range(len(syslist)):

        for it in trdata[key]:
            if it == syslist[ic]:
                ctr += 1
        tmatrix[0, ic] += ctr

        # reset sys call counter
        ctr = 0


np.save("../data/a16_atmatrix", tmatrix)
#with open('../data/a8_atmatrix.p', 'wb') as ha:
#    pickle.dump(tmatrix, ha)

print("System call statistics saved!")

