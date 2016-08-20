#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement an 1-class SVM classifier
# Iterate for all values of nu and plot ROC curve.

from __future__ import division
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# results file
resultsfile = '../data/d05-1csvm-v4-results.txt'
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

print("Loading adduser attack data.")
atdata = np.load('../data/b2_atmatrix.npy')

print("Loading validation data.")
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

# define 1-class SVM parameters
# cpar = 50
kernel1='sigmoid'
#nu1 = 0.5
gamma1 = 0.05
coef1 = 3

myprint("Starting cross validating an 1-class SVM model.")


# define multiple nu's to iterate!
nu = np.arange(0.1, 1.01, 0.1)


# ROC variables
fpr1 = []
acc1 = []


# iterate ...
for nu1 in nu:
    # train the model
    myprint("Parameters:\nKernel: {}\nnu: {}\ngamma: {}\ncoef: {}".format(kernel1, nu1, gamma1, coef1))
    model = OneClassSVM(kernel=kernel1, nu=nu1, gamma=gamma1, coef0=coef1, max_iter=20000)
    model.fit(trdata)

    # make predictions
    # Inliers are labeled 1, while outliers are labeled -1.

    pre1sum = sum(model.predict(trdata))
    pre1tot = trdata.shape[0]
    myprint("Performance on training data is          : {}".format(cperf(pre1sum, pre1tot, +1)))

    pre2sum = sum(model.predict(atdata))
    pre2tot = atdata.shape[0]
    met1 = cperf(pre2sum, pre2tot, -1)
    myprint("Attack detection Accuracy of the model is: {}".format(met1))

    pre3sum = sum(model.predict(vadata))
    pre3tot = vadata.shape[0]
    met2 = cperf(pre3sum, pre3tot, -1)
    myprint("False positive rate of the model is      : {}".format(met2))

    # add for roc
    acc1.append(met1)
    fpr1.append(met2)

with open(resultsfile, 'a') as ha:
    ha.write('\n')



# random classifier
rand = np.arange(0, 1.01, 0.2)

# This is the ROC curve

#plt.plot(fpr1, acc1, 'r', label='1class svm')

plt.scatter(fpr1, acc1, c='r', label='1class svm', marker = "D")
plt.plot(rand, rand, 'k', label='random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("1-class SVM classifier's ROC curves")
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
#plt.show() 
plt.savefig('../pictures/d05-1csvm-v4.eps')
plt.close()
