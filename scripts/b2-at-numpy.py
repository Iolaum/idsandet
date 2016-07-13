#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle
import numpy as np
import sys


print("Started processing attack data!")


with open('../data/attack.p', 'rb') as ha:
    atdata = pickle.load(ha)


# create numpy array to handle the data
# first number is lines, second is columns.
# 833 training data points
# 325 system calls to check their frequency.
tmatrix = np.zeros((746, 325))

# counter for line in array to be added
linectr = 0


syscalls = range(1, 326)

at1mat = np.empty((0, 325))  # Adduser atk
at2mat = np.empty((0, 325))  # Hydra ftp atk
at3mat = np.empty((0, 325))  # Hydra ssh atk
at4mat = np.empty((0, 325))  # Java Meterp atk
at5mat = np.empty((0, 325))  # Meterpreter atk
at6mat = np.empty((0, 325))  # Web Shell atk


for key in atdata:
    dummy1 = np.asarray(atdata[key])
    len1 = len(dummy1)

    # sys call counter
    ctr = 0

    # count system call frequency and fill Tr data matrix
    for ic in syscalls:
        for it in atdata[key]:
            if it == ic:
                ctr += 1
        tmatrix[linectr, (ic-1)] = ctr/len1

        # reset sys call counter
        ctr = 0

    dummyrow = tmatrix[linectr, :]
    dummyrow = dummyrow.reshape(1, 325)

    if key.find('UAD-Adduser') > -1:
        at1mat = np.append(at1mat, dummyrow, axis=0)

    elif key.find('UAD-Hydra-FTP') > -1:
        at2mat = np.append(at2mat, dummyrow, axis=0)

    elif key.find('UAD-Hydra-SSH') > -1:
        at3mat = np.append(at3mat, dummyrow, axis=0)

    elif key.find('UAD-Java-Meterpreter') > -1:
        at4mat = np.append(at4mat, dummyrow, axis=0)

    elif key.find('UAD-Meterpreter') > -1:
        at5mat = np.append(at5mat, dummyrow, axis=0)

    elif key.find('UAD-WS') > -1:
        at6mat = np.append(at6mat, dummyrow, axis=0)

    else:
        sys.exit("Attack method not identified!")

    # debug
    #print dummyrow
    #print dummyrow.shape
    #at1mat = np.append(at1mat, dummyrow, axis=0)
    #print at1mat
    #print at1mat.shape
    # exit()

    # debug
    #for it in tmatrix[0,:]:
    #    if it > 0:
    #        print it
    #break
    # debug

    # move to next row of tmatrix
    linectr += 1

    # create set of separate attack categories:

    # sample keys:
    # UAD-Hydra-FTP-4-2311 UAD-Adduser-1-2377 UAD-Hydra-SSH-3-2311.
    # find index of pre-last '-'
    # dumk = key.rfind('-', 0, key.rfind('-'))
    # dumk = key.rfind('-')
    # atset.add(key[0:dumk])

    # debug

    #if key[0:dumk] == 'UAD-Meterpreter':
    #    print("Debug me!")
    #    print key
    #    print atdata[key]
    # exit()

    # debug
    #if linectr == 1:
    #    break

# debug
#exit()

# atlist = list(atset)
# print atlist.sort()

with open('../data/b2_atmatrix.p', 'wb') as ha:
    pickle.dump(tmatrix, ha)

print("atmatrix saved as nd array with pickle.")

# creating files according categorized by attacks


with open('../data/b2_at1mat_adduser.p', 'wb') as ha:
    pickle.dump(at1mat, ha)

print("at1mat saved with shape {}".format(at1mat.shape))

with open('../data/b2_at2mat_hyftp.p', 'wb') as ha:
    pickle.dump(at2mat, ha)

print("at2mat saved with shape {}".format(at2mat.shape))

with open('../data/b2_at3mat_hyssh.p', 'wb') as ha:
    pickle.dump(at3mat, ha)

print("at3mat saved with shape {}".format(at3mat.shape))

with open('../data/b2_at4mat_javamet.p', 'wb') as ha:
    pickle.dump(at4mat, ha)

print("at4mat saved with shape {}".format(at4mat.shape))

with open('../data/b2_at5mat_meter.p', 'wb') as ha:
    pickle.dump(at5mat, ha)

print("at5mat saved with shape {}".format(at5mat.shape))

with open('../data/b2_at6mat_webshell.p', 'wb') as ha:
    pickle.dump(at6mat, ha)

print("at6mat saved with shape {}".format(at6mat.shape))

print("Validating data matrices!")
print("atmatrix shape is {}".format(tmatrix.shape))
atsums = (at1mat.shape[0] + at2mat.shape[0] + at3mat.shape[0] +
          at4mat.shape[0] + at5mat.shape[0] + at6mat.shape[0])
print("at*mat sums are {}".format((atsums, 325)))

print("Attack file data processed successfully!")
