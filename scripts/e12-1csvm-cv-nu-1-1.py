#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement an 1-class SVM classifier
# Iterate for all values of nu and plot ROC curve.

from __future__ import division
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit


# results file
resultsfile = '../data/e12-1csvm-cv-nu-v1-results.txt'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')


# calculate performance from sum of labels
def cperf(integer2, total, classlabel):
    integer = abs(integer2)
    perf = integer + ((total - integer)/2)
    result = perf / total
    if int(np.sign(integer2)) == classlabel:
        return result
    else:
        return 1-result


# Load results
# documentation: nu = np.arange(0.1, 1.01, 0.1)
with open('../data/e12-acc1-1.p', 'rb') as ha:
    acc1 = pickle.load(ha)

with open('../data/e12-fpr1-1.p', 'rb') as ha:
    fpr1 = pickle.load(ha)

with open('../data/e12-acc1er-1.p', 'rb') as ha:
    acc1er = pickle.load(ha)

with open('../data/e12-fpr1er-1.p', 'rb') as ha:
    fpr1er = pickle.load(ha)


print acc1
print acc1er
print fpr1
print fpr1er
#exit()

# random classifier
rand = np.arange(0, 1.01, 0.2)

# This is the ROC curve

#plt.plot(fpr1, acc1, 'r', label='1class svm')
#plt.scatter(fpr1, acc1, c='r', label='1class svm', marker = "D")

plt.errorbar(fpr1, acc1, xerr=fpr1er, yerr=acc1er, fmt='o')
plt.plot(rand, rand, 'k', label='random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("1-class SVM classifier's CV performance")
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.show() 
#plt.savefig('../pictures/e12-1csvm-v1-2.eps')
#plt.close()
