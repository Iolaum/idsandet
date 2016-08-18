#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle


print("Scipt starting!\nReading attack data file!")


with open('../data/1_attack.p', 'rb') as handle:
    atdata = pickle.load(handle)

# debug
print("Attack data size is {}".format(len(atdata)))

# determine max system call trace size
scsize = 0

for key in atdata:
    if len(atdata[key]) > scsize:
        scsize = len(atdata[key])
        # debug
        print("Current bigger system call trace is: {} from file {}.".format(scsize, key))

print("Script completed. Bigger system call trace is: {}.".format(scsize))


'''
$ ./a6-at-explore.py
Scipt starting!
Reading training data file!
Attack data size is 746
Current bigger system call trace is: 323 from file UAD-Hydra-FTP-4-2311.
Current bigger system call trace is: 1068 from file UAD-Adduser-1-2377.
Current bigger system call trace is: 1290 from file UAD-Hydra-SSH-3-2311.
Current bigger system call trace is: 2319 from file UAD-Hydra-SSH-1-1613.
Current bigger system call trace is: 2620 from file UAD-Adduser-9-17614.
Current bigger system call trace is: 2712 from file UAD-Hydra-SSH-2-2462.
Script completed. Bigger system call trace is: 2712.
'''
