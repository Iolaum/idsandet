#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Perform Cross Validation with SVMs

from __future__ import division
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


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
resultsfile = '../data/e09-svm-v2-results.txt'

# custom print function
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')





# # Load Data!
print("Loading trainind data.")
trdat = np.load('../data/e06a-trdata.npy')

#print trdat.shape
#exit()

print("Loading attack data.")
atdat = np.load('../data/e06c-atdata.npy')

print("Loading validation data.")
vadat = np.load('../data/e06b-vadata.npy')


# Rescale the data!
alldat = np.concatenate((trdat, vadat, atdat), axis=0)
preproc = MinMaxScaler(copy=False)
preproc.fit(alldat)


del alldat

trdat = preproc.transform(trdat)
vadat = preproc.transform(vadat)
atdat = preproc.transform(atdat)

del preproc

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
cpar = 32
model = SVC(C=cpar, kernel='linear', max_iter=20000, verbose=True, class_weight='balanced')

myprint("Starting Cross Validation on a SVM model with linear kernel and C={}.".format(cpar))

# y = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
# print y

skf = StratifiedKFold(ydat, 8, random_state = 666)

#print skf

# iteration counter
ctr = 1

# performance metrics !!
precision = []
fallout = []

for train_index, test_index in skf:
    print("Iteration {}: Train set #{} - Validation set #{}".format(ctr, train_index.shape, test_index.shape))
    # print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        
    print("Building training data.")

    trxdat = np.zeros((len(train_index), 3792))
    trydat = np.zeros(len(train_index))

    for it1, val1 in enumerate(train_index):
        trxdat[it1, :] = xdat[val1, :].reshape(1,3792)
        trydat[it1] = ydat[val1]

    print("Building validation data.")

    tsxdat = np.zeros((len(test_index), 3792))
    tsydat = np.zeros(len(test_index))

    for it1, val1 in enumerate(test_index):
        tsxdat[it1, :] = xdat[val1, :].reshape(1,3792)
        tsydat[it1] = ydat[val1]


    # train the model
    print("Starting training an SVM model with linear kernel and C={}.".format(cpar))
    model.fit(trxdat, trydat)

    # make predictions
    pre = model.predict(tsxdat)
    
    #print tsydat.shape
    #print pre.shape

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
