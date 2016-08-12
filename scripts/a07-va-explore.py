#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle


print("Scipt starting!\nReading training data file!")


with open('../data/validation.p', 'rb') as handle:
    vadata = pickle.load(handle)

# debug
print("Attack data size is {}".format(len(vadata)))

# determine max system call trace size
scsize = 0

for key in vadata:
    if len(vadata[key]) > scsize:
        scsize = len(vadata[key])
        # debug
        print("Current bigger system call trace is: {} from file {}.".format(scsize, key))

print("Script completed. Bigger system call trace is: {}.".format(scsize))


'''
$ ./a7-va-explore.py 
Scipt starting!
Reading training data file!
Attack data size is 4372
Current bigger system call trace is: 262 from file UVD-1539.
Current bigger system call trace is: 564 from file UVD-3742.
Current bigger system call trace is: 781 from file UVD-3748.
Current bigger system call trace is: 997 from file UVD-3749.
Current bigger system call trace is: 1443 from file UVD-0924.
Current bigger system call trace is: 2252 from file UVD-0920.
Current bigger system call trace is: 2410 from file UVD-2017.
Current bigger system call trace is: 3063 from file UVD-3836.
Current bigger system call trace is: 3768 from file UVD-2557.
Current bigger system call trace is: 4000 from file UVD-0224.
Current bigger system call trace is: 4494 from file UVD-0335.
Script completed. Bigger system call trace is: 4494.
'''
