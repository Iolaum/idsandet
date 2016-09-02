#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# PLOT Perform Recursive Feature Elimination

from __future__ import division
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
np.set_printoptions(threshold=np.inf)


# custom scoring function
def myscore(estimator, xval, ytrue):
    # #debug
    print("Starting Scoring function.\n{} features left".format(xval.shape[1]))

    # make predictions
    pre = estimator.predict(xval)

    #print("TEST")
    cmat = confusion_matrix(ytrue, pre)
    #print("Confusion matrix is \n{}".format(cmat))

    # Attack detection accuracy || precision rate = TP/(TP+FP)
    # (1,1)/[(1,1)+(1,0)]
    met1 = cmat[1,1]/(cmat[1,0]+cmat[1,1])
    print("Attack detection Accuracy of the model is {}".format(met1))

    # False Positive Rate || ... = FN/(FN+TN)
    # (0,1)/[(0,0)+(0,1)]
    met2 = cmat[0,1]/(cmat[0,0]+cmat[0,1])
    print("False positive rate of the model is       {}".format(met2))

    #print(met1 - met2)
    return (met1 - met2)


# results file
resultsfile = '../data/f04-rfe-results.txt'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')


# load frequency space results

with open('../data/d07-rfeobj-v1-1.p', 'rb') as ha:
    rfecv = pickle.load(ha)


# load syscall list
with open('../data/a14-sys-list.p', 'rb') as ha:
    syslist = pickle.load(ha)


myprint("Number of more informative system calls: %d" % rfecv.n_features_)

#print("System call ranking is:")
#print len(rfecv.ranking_)
#print len(syslist)


for key, value in enumerate(syslist):
    if rfecv.ranking_[key] == 1:
        myprint(str(value))







