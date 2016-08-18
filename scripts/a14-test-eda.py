#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Get number of all system calls

from __future__ import division
import pickle
import numpy as np


# load sequencis of training data
with open('../data/1_training.p', 'rb') as ha:
    trdata = pickle.load(ha)

# system calls set
syset = set()

print("Populating data object.")

# populate ...
for key, value in trdata.iteritems():

    for it1 in range(len(value)):
        syset.add(value[it1])
    #exit()

# load sequencis of validation data
with open('../data/1_validation.p', 'rb') as ha:
    trdata = pickle.load(ha)

# populate ...
for key, value in trdata.iteritems():

    for it1 in range(len(value)):
        syset.add(value[it1])

# load sequencis of attack data
with open('../data/1_attack.p', 'rb') as ha:
    trdata = pickle.load(ha)

# populate ...
for key, value in trdata.iteritems():

    for it1 in range(len(value)):
        syset.add(value[it1])
    #exit()
print("There are {} system calls.".format(len(syset)))

with open('../data/a14-sys-set.p', 'wb') as ha:
    pickle.dump(syset, ha)

syslist = list(syset)
syslist.sort

with open('../data/a14-sys-list.p', 'wb') as ha:
    pickle.dump(syslist, ha)

print syslist

'''
[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 19, 20, 21, 22, 26, 27, 30, 33, 37, 38, 39, 40, 41, 42, 43, 45, 54, 57, 60, 61, 63, 64, 65, 66, 75, 77, 78, 79, 83, 85, 90, 91, 93, 94, 96, 97, 99, 102, 104, 110, 111, 114, 116, 117, 118, 119, 120, 122, 124, 125, 128, 132, 133, 136, 140, 141, 142, 143, 144, 146, 148, 150, 151, 154, 155, 156, 157, 158, 159, 160, 162, 163, 168, 172, 173, 174, 175, 176, 177, 179, 180, 181, 183, 184, 185, 186, 187, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 219, 220, 221, 224, 226, 228, 229, 230, 231, 233, 234, 240, 242, 243, 252, 254, 255, 256, 258, 259, 260, 264, 265, 266, 268, 269, 270, 272, 289, 292, 293, 295, 296, 298, 300, 301, 306, 307, 308, 309, 311, 314, 320, 322, 324, 328, 331, 332, 340]
'''
