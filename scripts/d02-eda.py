#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Exploratory Data Analysis
# Deprecated!
# Incorrect!!

from __future__ import division
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# results file
resultsfile = '../data/d02-eda-results.txt'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

# load dataset

with open('../data/b1_trmatrix.p', 'rb') as ha:
    trdata = pickle.load(ha)

print("Loaded training data.   trdata shape is {}".format(trdata.shape))

with open('../data/b2_atmatrix.p', 'rb') as ha:
    atdata = pickle.load(ha)

print("Loaded attack data.     atdata shape is {}".format(atdata.shape))

with open('../data/b3_vamatrix.p', 'rb') as ha:
    vadata = pickle.load(ha)

print("Loaded validation data. vadata shape is {}".format(vadata.shape))


alldata = np.concatenate((trdata, vadata, atdata), axis=0)

# delete misc data
# del trdata
# del atdata
# del vadata

print("Combined all the dataset into a matrix with dimensions {}".format(alldata.shape))

datasums =  np.sum(alldata, axis=0)

# debug
# testsum =  np.sum(alldata, axis=1)
# for it in range(20):
#     print testsum[it]
# exit()
# some frequencies summ up to less than 1 !
# ~0.2% overall round error ?!

print("Sum of frequencies is: {}".format(np.sum(datasums)))

# count nonzero system call frequencies
print np.count_nonzero(datasums)


# find zero freq calls
indx = []

for it in range(325):
    if datasums[it]==0:
        indx.append(it)

print("Found {} zero frequency entries".format(len(indx)))

# sodr in descending order so deleting is consistent
indx.sort(reverse=True)

# sys call list
syscalls = np.array(range(1, 326))

# delete uncalled system calls
for it in indx:
    trdata = np.delete(trdata, it, 1)
    atdata = np.delete(atdata, it, 1)
    vadata = np.delete(vadata, it, 1)
    syscalls = np.delete(syscalls, it, 0)

## debug
#print syscalls

with open('../data/d2_trmatrix.p', 'wb') as ha:
    pickle.dump(trdata, ha)

with open('../data/d2_atmatrix.p', 'wb') as ha:
    pickle.dump(atdata, ha)

with open('../data/d2_vamatrix.p', 'wb') as ha:
    pickle.dump(vadata, ha)

with open('../data/d2_syscalls.p', 'wb') as ha:
    pickle.dump(syscalls, ha)

print("Data saved after deleting zero frequencies.")

alldata = np.concatenate((trdata, vadata, atdata), axis=0)
datasums =  np.sum(alldata, axis=0)
print("Verification: Sum of frequencies is: {}".format(np.sum(datasums)))

'''
[  1   3   4   5   6   7   8   9  10  11  12  13  15  19  20  21  22  26
  27  30  33  37  38  39  40  41  42  43  45  54  57  60  61  63  64  65
  66  75  77  78  79  83  85  90  91  93  94  96  97  99 102 104 110 111
 114 116 117 118 119 120 122 124 125 128 132 133 136 140 141 142 143 144
 146 148 150 151 154 155 156 157 158 159 160 162 163 168 172 173 174 175
 176 177 179 180 181 183 184 185 186 187 190 191 192 194 195 196 197 198
 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216
 219 220 221 224 226 228 229 230 231 233 234 240 242 243 252 254 255 256
 258 259 260 264 265 266 268 269 270 272 289 292 293 295 296 298 300 301
 306 307 308 309 311 314 320 322 324]
'''
