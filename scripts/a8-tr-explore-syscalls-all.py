#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

import pickle
import numpy as np
from os.path import exists


print("Started processing training data!")


# use model if it exits

if exists('../data/a8_trmatrix.p'):
    print('Loading system call statistics.')
    with open('../data/a8_trmatrix.p', 'rb') as ha:
        tmatrix = pickle.load(ha)
else:

    with open('../data/1_training.p', 'rb') as ha:
        trdata = pickle.load(ha)

    # create numpy array to handle the data
    # first number is lines, second is columns.
    # 833 training data points
    # 325 system calls to check their frequency.
    tmatrix = np.zeros((1, 325))


    syscalls = range(1, 326)

    for key in trdata:
        dummy1 = np.asarray(trdata[key])


        # sys call counter
        ctr = 0

        # count system call frequency and fill Tr data matrix
        for ic in syscalls:
            for it in trdata[key]:
                if it == ic:
                    ctr += 1
            tmatrix[0, (ic-1)] += ctr

            # reset sys call counter
            ctr = 0

        # debug
        #for it in tmatrix[0,:]:
        #    if it > 0:
        #        print it
        #break
        # debug


        # debug
        #if linectr == 5:
        #    break


    with open('../data/a8_trmatrix.p', 'wb') as ha:
        pickle.dump(tmatrix, ha)

    print("System call statistics saved!")

print tmatrix
print ("Number of training set system calls that are not called is: {}".format(np.count_nonzero(tmatrix)))

'''
Number of training set system calls that are not called is: 147
'''
