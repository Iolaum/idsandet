#!/usr/bin/env python
# -*- coding: utf-8 -*-

# stratified kfold CV with linear SVM


from __future__ import division
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# results file
resultsfile = '../data/d01-svm-cv-1-results.txt'
# custom print function to also save runs on text file
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


# load dataset

with open('../data/d2_trmatrix.p', 'rb') as ha:
    trdata = pickle.load(ha)

print("Loaded training data.   trdata shape is {}".format(trdata.shape))

with open('../data/d2_atmatrix.p', 'rb') as ha:
    atdata = pickle.load(ha)

print("Loaded attack data.     atdata shape is {}".format(atdata.shape))

with open('../data/d2_vamatrix.p', 'rb') as ha:
    vadata = pickle.load(ha)

print("Loaded validation data. vadata shape is {}".format(vadata.shape))


tvdata = np.concatenate((trdata, vadata), axis=0)
# delete unneeded data
del trdata
del vadata

# create classification labels
l1 = np.zeros(tvdata.shape[0])
l2 = np.ones(atdata.shape[0])
# print l2.shape


# split train set
xtr, xts, ytr, yts = train_test_split(np.concatenate((tvdata, atdata), axis=0), np.concatenate((l1,l2), axis=0), 
    test_size=0.3, random_state=69) #, stratify=np.concatenate((l1,l2), axis=0))

# print np.sum(ytr)

# debug
# print xtr.shape
# print ytr.shape


# define SVM parameters
cpar = 50
model = SVC(C=cpar, kernel='linear', max_iter=20000, verbose=True, class_weight='balanced') # 

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel


# train the model
myprint("Starting training an SVM model with linear kernel and C={}.".format(cpar))
model.fit(xtr, ytr)

# make predictions
pre = model.predict(xts)


cmat = confusion_matrix(yts, pre)

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
