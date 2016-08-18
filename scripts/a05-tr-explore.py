#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a 
# training data dictionary of lists.

from __future__ import division
import pickle


print("Scipt starting!\nReading training data file!")


with open('../data/1_training.p', 'rb') as handle:
  trdata = pickle.load(handle)

# debug
print("Training data size is {}".format(len(trdata)))

# determine max system call trace size
scsize = 0

for key in trdata:
    if len(trdata[key]) > scsize:
        scsize = len(trdata[key])
        # debug
        print("Current bigger system call trace is: {} from file {}.".format(scsize, key))

print("Script completed. Bigger system call trace is: {}.".format(scsize))


'''
$ ./a1-tr-explore.py 
Scipt starting!
Reading training data file!
Training data size is 833
Current bigger system call trace is: 110 from file UTD-0763.
Current bigger system call trace is: 513 from file UTD-0762.
Current bigger system call trace is: 1508 from file UTD-0766.
Current bigger system call trace is: 1769 from file UTD-0279.
Current bigger system call trace is: 1783 from file UTD-0388.
Current bigger system call trace is: 1830 from file UTD-0246.
Current bigger system call trace is: 1881 from file UTD-0562.
Current bigger system call trace is: 2948 from file UTD-0134.
Script completed. Bigger system call trace is: 2948.
'''
