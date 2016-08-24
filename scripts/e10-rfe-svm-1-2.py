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



with open('../data/e10-rfeobj-v1-1.p', 'rb') as ha:
    rfecv = pickle.load(ha)

print("Optimal number of features : %d" % rfecv.n_features_)
#exit()

# create feature x-axis index
xra = []
for it1 in rfecv.grid_scores_:
    if len(xra) == 0:
        xra.append(1)
    elif len(xra) == 1:
        xra.append(24)
    else:
        xra.append(xra[len(xra)-1]+24)


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("# Features")
plt.ylabel("Scorer")
plt.title('Performance of Linear SVM classifier per # of features.')
plt.plot(xra, rfecv.grid_scores_)
#plt.show()
plt.savefig('../pictures/e10-rfe-svm-1.eps')
plt.close()


