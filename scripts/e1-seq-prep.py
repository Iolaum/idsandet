#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Preparing a short sequence feature engineering.

from __future__ import division
import pickle
import numpy as np


with open('../data/d2_syscalls.p', 'rb') as ha:
    syscalls = pickle.load(ha)

# dictionary setup !
#
# key-value-1
# syscal sequence : integer pointer
# (33, 79): 12
#
# key-value-2
# integer pointer : syscall sequence
# 12: (33, 79)

#create empty dict
sysdict = {}

# pointer for later reference to system calls.
pointer = 0

for it1 in syscalls:
    for it2 in syscalls:
        # debug
        # print (it1,it2)
        sysdict[(it1, it2)] = pointer
        sysdict[pointer] = (it1, it2)
        pointer += 1

with open('../data/e1-sys_seq-dict.p', 'wb') as ha:
    pickle.dump(sysdict, ha)

print("Total number of sequencies is {}".format(pointer+1))
# Total number of sequencies is 29242

print("System calls 2-sequence dictionary created.")
