#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Stratified Kfold CV with linear SVM

from __future__ import division
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix


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

# Attack detection accuracy || precision rate = TP/(TP+FP)
# (1,1)/[(1,1)+(1,0)]

# False Positive Rate || fall out = FN/(FN+TN)
# (0,1)/[(0,0)+(0,1)]

# custom print function to also save runs on text file
resultsfile = '../data/d02-1-results.txt'

# custom print function
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')



#myprint("Starting a linear SVM classifier.")

# # Load Data!
print("Loading trainind data.")
trdat = np.load('../data/b1_trmatrix.npy')

print("Loading adduser attack data.")
atdat = np.load('../data/b2_atmatrix.npy')

print("Loading validation data.")
vadat = np.load('../data/b3_vamatrix.npy')


tvdat = np.concatenate((trdat, vadat), axis=0)
# delete unneeded data
del trdat
del vadat

# create classification labels
l1 = np.zeros(tvdat.shape[0])
l2 = np.ones(atdat.shape[0])
# print l2.shape

xdat = np.concatenate((tvdat, atdat), axis=0)
ydat = np.concatenate((l1,l2), axis=0)

# delete undeleted data
del tvdat
del atdat
del l1
del l2


# shuffle data first
xdat, ydat = shuffle (xdat, ydat, random_state=96)

# define SVM parameters
cpar = 256
model = SVC(C=cpar, kernel='linear', max_iter=50000, verbose=True, class_weight='balanced')

myprint("Starting Stratified kfold CV with a linear SVM and C={}.".format(cpar))


# y = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
# print y

# skf
skf = StratifiedKFold(ydat[0:1000], 8, random_state = 666)

#print skf

# iteration counter
ctr = 1

# performance metrics !!
precision = []
fallout = []

for train_index, test_index in skf:
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


       
    #print nxdat.shape
    #print nydat.shape
    #exit()

    # train the model
    print("Starting training an SVM model with linear kernel and C={}.".format(cpar))
    model.fit(trxdat, trydat)

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
