#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Add system call frequency

import pickle
import numpy as np
from os.path import exists


print("Started processing training data!")


# use model if it exits

if exists('../data/a10_vamatrix.p'):
    print('Loading system call statistics.')
    with open('../data/a10_vamatrix.p', 'rb') as ha:
        tmatrix = pickle.load(ha)
else:

    with open('../data/b3_vamatrix.p', 'rb') as ha:
        trdata = pickle.load(ha)

    # create numpy array to handle the data
    # first number is lines, second is columns.
    # 833 training data points
    # 325 system calls to check their frequency.
    tmatrix = np.zeros((1, 325))


    syscalls = range(1, 326)


    #iterate over all elements and add rel frequencies:
    
    # #_lines
    nlin = trdata.shape[0]

    linit = range(nlin)


    # iterate over all data points    
    for it1 in linit:
        
        # iterate over all sys calls
        for it2 in range(325):
    
            # add rel frequencies
            tmatrix[0, it2] += trdata[it1, it2]

    # divide by total number of points
    tmatrix = np.divide(tmatrix, nlin)

    # save
    with open('../data/a10_vamatrix.p', 'wb') as ha:
        pickle.dump(tmatrix, ha)

    print("System call statistics saved!")

print tmatrix
print ("Number of attack set system calls that are not called is: {}".format(np.count_nonzero(tmatrix)))

'''
Number of validation set system calls that are not called is: 165
'''
