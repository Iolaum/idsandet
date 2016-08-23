#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Train an SVM model.

from __future__ import division
import numpy as np
from sklearn.svm import SVC
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

# results file
resultsfile = '../data/e08-svm-v2-results.txt'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')


# load data
xtr = np.load('../data/e07a-xtr.npy')
ytr = np.load('../data/e07b-ytr.npy')
xts = np.load('../data/e07c-xts.npy')
yts = np.load('../data/e07d-yts.npy')


# Rescale the data!
alldata = np.concatenate((xtr, xts), axis=0)
preproc = MinMaxScaler(copy=False)
preproc.fit(alldata)

del alldata

xtr = preproc.transform(xtr)
xts = preproc.transform(xts)





# define SVM parameters
cpar = 100
model = SVC(C=cpar, kernel='linear', max_iter=25000, verbose=True, class_weight='balanced') # 

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

# train the model
myprint("Starting training an SVM model with linear kernel and C={}.".format(cpar))
model.fit(xtr, ytr)

# make predictions
pre = model.predict(xts)


cmat = confusion_matrix(yts, pre)

# print cmat

# debug
# print cmat
#print yts.shape

# Validation set size
#valsz = yts.shape[0]
#print valsz


# Attack detection accuracy
# (1,1)/[(1,1)+(1,0)]
myprint("Attack detection Accuracy of the model is {}".format(cmat[1,1]/(cmat[1,0]+cmat[1,1])))

# False Positive Rate
# (0,1)/[(0,0)+(0,1)]
myprint("False positive rate of the model is {}".format(cmat[0,1]/(cmat[0,0]+cmat[0,1])))


with open(resultsfile, 'a') as ha:
    ha.write('\n')

