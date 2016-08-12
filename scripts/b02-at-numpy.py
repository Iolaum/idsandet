#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load attack data and count system call frequencies

from __future__ import division
import pickle
import numpy as np
import sys


print("Started processing attack data!")

# load attack data
with open('../data/1_attack.p', 'rb') as ha:
    atdata = pickle.load(ha)

# load non-zero sys calls
with open('../data/a14-sys-list.p', 'rb') as ha:
    syslist = pickle.load(ha)

# create numpy array to handle the data
# first number is lines, second is columns.
tmatrix = np.zeros((746, 175))

# counter for line in array to be added
linectr = 0


at1mat = np.empty((0, 175))  # Adduser atk
at2mat = np.empty((0, 175))  # Hydra ftp atk
at3mat = np.empty((0, 175))  # Hydra ssh atk
at4mat = np.empty((0, 175))  # Java Meterp atk
at5mat = np.empty((0, 175))  # Meterpreter atk
at6mat = np.empty((0, 175))  # Web Shell atk


for key in atdata:
    dummy1 = np.asarray(atdata[key])
    len1 = len(dummy1)

    # sys call counter
    ctr = 0

    # count system call frequency and fill Tr data matrix
    for ic in range(len(syslist)):
        for it in atdata[key]:
            if it == syslist[ic]:
                ctr += 1
        tmatrix[linectr, ic] = ctr/len1

        # reset sys call counter
        ctr = 0

    dummyrow = tmatrix[linectr, :]
    #debug
    #print dummyrow.shape
    #print dummyrow
    
    dummyrow = dummyrow.reshape(1, 175)
    #print dummyrow
    #exit()

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


    # move to next row of tmatrix
    linectr += 1

np.save("../data/b2_atmatrix", tmatrix)

print("atmatrix saved as npy array with numpy.")

# creating files according categorized by attacks

np.save("../data/b2_at1mat_adduser", at1mat)
print("at1mat saved with shape {}".format(at1mat.shape))

np.save("../data/b2_at2mat_hyftp", at2mat)
print("at2mat saved with shape {}".format(at2mat.shape))

np.save("../data/b2_at3mat_hyssh", at3mat)
print("at3mat saved with shape {}".format(at3mat.shape))

np.save("../data/b2_at4mat_javamet", at4mat)
print("at4mat saved with shape {}".format(at4mat.shape))

np.save("../data/b2_at5mat_meter", at5mat)
print("at5mat saved with shape {}".format(at5mat.shape))

np.save("../data/b2_at6mat_webshell", at6mat)
print("at6mat saved with shape {}".format(at6mat.shape))

print("Validating data matrices!")
print("atmatrix shape is {}".format(tmatrix.shape))
atsums = (at1mat.shape[0] + at2mat.shape[0] + at3mat.shape[0] +
          at4mat.shape[0] + at5mat.shape[0] + at6mat.shape[0])
print("at*mat sums are {}".format((atsums, at1mat.shape[1])))

print("Attack file data processed successfully!")
