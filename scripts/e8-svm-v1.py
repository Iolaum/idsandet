#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Train an SVM model.

from __future__ import division
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# load data
xtr = np.load('../data/e7a-xtr.npy')
ytr = np.load('../data/e7b-ytr.npy')
xts = np.load('../data/e7c-xts.npy')
yts = np.load('../data/e7d-yts.npy')

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
resultsfile = '../data/e8-svm-v1-results.txt'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

# debug
# print ytr[1:10]
# print yts[1:10]



# define SVM parameters
cpar = 50
model = SVC(C=cpar, kernel='linear', max_iter=5000, verbose=True, class_weight='balanced') # 

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

