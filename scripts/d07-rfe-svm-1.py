#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Perform Recursive Feature Elimination

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


# custom print function to also save runs on text file
resultsfile = '../data/d07-rfe-svm-v1-results.txt'

# custom print function
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

# confusion_matrix:
#
# Positive = Attack !
#
#      Predicted  #     Predicted
# Ac   0,0 | 0,1  # Ac   TN | FN
# tu   ----|----  # tu   ---|---
# al   1,0 | 1,1  # al   FP | TP
#
# (Actual, Predicted)


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


# # Load Data!
print("Starting Recursive Feature Elimination.")

trdata = np.load('../data/b1_trmatrix.npy')
atdata = np.load('../data/b2_atmatrix.npy')
vadata = np.load('../data/b3_vamatrix.npy')

print("Loaded training data.   trdata shape is {}".format(trdata.shape))
print("Loaded attack data.     atdata shape is {}".format(atdata.shape))
print("Loaded validation data. vadata shape is {}".format(vadata.shape))


# Rescale the data!
alldata = np.concatenate((trdata, atdata, vadata), axis=0)
preproc = MinMaxScaler(copy=False)
preproc.fit(alldata)

del alldata

trdata = preproc.transform(trdata)
atdata = preproc.transform(atdata)
vadata = preproc.transform(vadata)

tvdata = np.concatenate((trdata, vadata), axis=0)
# delete unneeded data
del trdata
del vadata

# create classification labels
l1 = np.zeros(tvdata.shape[0])
l2 = np.ones(atdata.shape[0])
# print l2.shape

xdat = np.concatenate((tvdata, atdata), axis=0)
ydat = np.concatenate((l1,l2), axis=0)

# delete undeleted data
del tvdata
del atdata
del l1
del l2


# shuffle data first
xdat, ydat = shuffle(xdat, ydat, random_state=96)

# define SVM parameters
cpar = 1
kernel1 = 'linear'
model = SVC(C=cpar, kernel=kernel1, max_iter=5000, verbose=False, class_weight='balanced')
 

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

# Specify RFE parameters
step1 = 1

rfecv = RFECV(estimator=model, step=step1,
    cv=StratifiedShuffleSplit(ydat, n_iter=6, test_size=0.5), 
    scoring=myscore)

##
myprint("Starting Recursive Feature Elimination")
myprint("Parameters:\nKernel: {} SVM\nElimination Step: {}\nRegularisation (C): {}".format(kernel1, step1, cpar))

# Perform RFE (...)
rfecv.fit(xdat, ydat)

myprint("Optimal number of features : %d" % rfecv.n_features_)


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("# Features")
plt.ylabel("Performance")
plt.title('Performance of Linear SVM classifier per # of features.')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()
plt.savefig('../pictures/d07-rfe-svm-1.eps')
plt.close()

with open('../data/d07-rfeobj-v1-1.p', 'wb') as ha:
    pickle.dump(rfecv, ha)

with open(resultsfile, 'a') as ha:
    ha.write('\n')

