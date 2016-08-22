#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement an 1-class SVM classifier

from __future__ import division
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


# results file
resultsfile = '../data/d06-cv-1csvm-v2-results.txt'
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





# # Load Data!
print("Loading trainind data.")
trdata = np.load('../data/b1_trmatrix.npy')

print("Loading attack data.")
atdata = np.load('../data/b2_atmatrix.npy')

print("Loading validation data.")
vadata = np.load('../data/b3_vamatrix.npy')


print("Loaded training data.   trdata shape is {}".format(trdata.shape))
print("Loaded attack data.     atdata shape is {}".format(atdata.shape))
print("Loaded validation data. vadata shape is {}".format(vadata.shape))


tvdata = np.concatenate((trdata, vadata), axis=0)
# delete unneeded data
del trdata
del vadata

# create classification labels
# Inliers are labeled 1, while outliers are labeled -1.
l1 = np.ones(tvdata.shape[0])
l2 = - np.ones(atdata.shape[0])
# print l2.shape

xdat = np.concatenate((tvdata, atdata), axis=0)
ydat = np.concatenate((l1,l2), axis=0)

# delete undeleted data
del tvdata
del atdata
del l1
del l2


# shuffle data first
xdat, ydat = shuffle (xdat, ydat, random_state=96)

# Rescale the data! - fit + transform
preproc = MinMaxScaler(copy=False)
xdat = preproc.fit_transform(xdat)


# define 1-class SVM parameters
# cpar = 50
kernel1='sigmoid'
nu1 = 0.45
gamma1 = 0.05
coef1 = 3
model = OneClassSVM(kernel=kernel1, nu=nu1, gamma=gamma1, coef0=coef1, max_iter=20000, verbose=True)

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel


myprint("Starting training an 1-class SVM model.")
myprint("Parameters:\nKernel: {}\nnu: {}\ngamma: {}\ncoef: {}".format(kernel1, nu1, gamma1, coef1))

# CV
skf = StratifiedShuffleSplit(ydat, n_iter=10, test_size=0.5)

# iteration counter
ctr = 1


# performance metrics !!
precision = []
fallout = []


for train_index, test_index in skf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    print("Iteration {}: Train set #{} - Validation set #{}".format(ctr, train_index.shape, test_index.shape))
    # print("TRAIN:", train_index.shape, "TEST:", test_index.shape)

    for it1 in train_index:
        try:
            trxdat = np.concatenate((trxdat, xdat[it1, :].reshape(1,175)), axis=0)
            trydat = np.append(trydat, ydat[it1])
        except NameError:
            trxdat = xdat[it1, :].reshape(1,175)
            trydat = ydat[it1]


    for it1 in test_index:
        try:
            tsxdat = np.concatenate((tsxdat, xdat[it1, :].reshape(1,175)), axis=0)
            tsydat = np.append(tsydat, ydat[it1])
        except NameError:
            tsxdat = xdat[it1, :].reshape(1,175)
            tsydat = ydat[it1]

    # train the model
    model.fit(trxdat)

    # make predictions
    pre = model.predict(tsxdat)

    cmat = confusion_matrix(tsydat, pre)
    #print cmat

    # Attack detection accuracy || precision rate = TP/(TP+FP)
    # (1,1)/[(1,1)+(1,0)]
    met1 = cmat[1,1]/(cmat[1,0]+cmat[1,1])
    print("Attack detection Accuracy of the model is {}".format(met1))

    # False Positive Rate || ... = FN/(FN+TN)
    # (0,1)/[(0,0)+(0,1)]
    met2 = cmat[0,1]/(cmat[0,0]+cmat[0,1])
    print("False positive rate of the model is       {}".format(met2))

    ctr += 1
    precision.append(met1)
    fallout.append(met2)

    del trxdat
    del trydat
    del tsxdat
    del tsydat


myprint("Average precision is {} +/- {}".format(np.mean(precision), np.std(precision)))
myprint("Average fall out  is {} +/- {}".format(np.mean(fallout), np.std(fallout)))

with open(resultsfile, 'a') as ha:
    ha.write('\n')
