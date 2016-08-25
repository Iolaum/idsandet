#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement an 1-class SVM classifier

from __future__ import division
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


# results file
resultsfile = '../data/e11-1csvm-v1-results.txt'
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
print("Starting One class SVM exploration.")

trdata = np.load('../data/e06a-trdata.npy')
atdata = np.load('../data/e06c-atdata.npy')
vadata = np.load('../data/e06b-vadata.npy')


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

# merge to re-split !
tvdata = np.concatenate((trdata, vadata), axis=0)
del trdata
del vadata

# shuffle data first
tvdata = shuffle(tvdata) # , random_state=96


# resplit !!
ind1 = int(tvdata.shape[0]/2)
trdata = tvdata[:ind1,:]
vadata = tvdata[ind1:, :]

# define 1-class SVM parameters
# cpar = 50
kernel1='sigmoid'
# nu: bound for training mis classification
nu1 = 0.5 
# gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
# If gamma is ‘auto’ then 1/n_features will be used instead.
gamma1 = 0.005
# coef1: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
coef1 = 0.5
model = OneClassSVM(kernel=kernel1, nu=nu1, gamma=gamma1, coef0=coef1, max_iter=20000, verbose=True)

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel


# train the model
# myprint("Starting training an 1-class SVM model with linear kernel and C={}.".format(cpar))
myprint("Starting training an 1-class SVM model.")
myprint("Parameters:\nKernel: {}\nnu: {}\ngamma: {}\ncoef: {}".format(kernel1, nu1, gamma1, coef1))
model.fit(trdata)

# make predictions
# Inliers are labeled 1, while outliers are labeled -1.

pre1sum = sum(model.predict(trdata))
pre1tot = trdata.shape[0]
#print("Verifying. Prediction sum on trdata is: {}".format(pre1sum))
myprint("Performance on training data is {}".format(cperf(pre1sum, pre1tot, +1)))

pre2sum = sum(model.predict(atdata))
pre2tot = atdata.shape[0]
acc1 = cperf(pre2sum, pre2tot, -1)
myprint("Attack detection Accuracy of the model is {}".format(acc1))

pre3sum = sum(model.predict(vadata))
pre3tot = vadata.shape[0]
fpr1 = cperf(pre3sum, pre3tot, -1)
myprint("False positive rate of the model is {}".format(fpr1))
myprint("Scorer performance is: {}".format(acc1 - fpr1))

with open(resultsfile, 'a') as ha:
    ha.write('\n')
