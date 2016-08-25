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
from sklearn.cross_validation import ShuffleSplit


# results file
resultsfile = '../data/e12-1csvm-cv-nu-v1-results.txt'
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

# define 1-class SVM parameters
# cpar = 50
kernel1='sigmoid'
#nu1 = 0.5
gamma1 = 0.05
coef1 = 3

myprint("Starting cross validating an 1-class SVM model.")
myprint("Parameters:\nKernel: {}\ngamma: {}\ncoef: {}".format(kernel1, gamma1, coef1))


# CV
cv = ShuffleSplit(tvdata.shape[0], n_iter=10, test_size=0.5)

# define multiple nu's to iterate!
nu = np.arange(0.1, 1.01, 0.1)


# performance metrics !!
acc1 = []
fpr1 = []

# standard deviations for errors
acc1er = []
fpr1er = []

# length for resize
alen = tvdata.shape[1]

# iterate ...
for nu1 in nu:

    # performance metrics !! - dummy
    dacc1 = []
    dfpr1 = []

    # iteration counter
    ctr = 1

    for train_index, test_index in cv:
        #print("TRAIN:", train_index, "TEST:", test_index)
        print("Iteration {}, nu= {}: Train set #{} - Validation set #{}".format(ctr, nu1, train_index.shape, test_index.shape))
        # print("TRAIN:", train_index.shape, "TEST:", test_index.shape)


        print("Building training data.")

        trdata = np.zeros((len(train_index), alen))

        for it1, val1 in enumerate(train_index):
            trdata[it1, :] = tvdata[val1, :].reshape(1,alen)
 
        print("Building validation data.")
  
        vadata = np.zeros((len(test_index), alen))

        for it1, val1 in enumerate(test_index):
            vadata[it1, :] = tvdata[val1, :].reshape(1, alen)


        # train the model
        print("Starting Training!")
        model = OneClassSVM(kernel=kernel1, nu=nu1, gamma=gamma1, coef0=coef1, max_iter=20000)
        model.fit(trdata)

        # make predictions
        # Inliers are labeled 1, while outliers are labeled -1.

        pre1sum = sum(model.predict(trdata))
        pre1tot = trdata.shape[0]
        print("Performance on training data is          : {}".format(cperf(pre1sum, pre1tot, +1)))

        pre2sum = sum(model.predict(atdata))
        pre2tot = atdata.shape[0]
        met1 = cperf(pre2sum, pre2tot, -1)
        print("Attack detection Accuracy of the model is: {}".format(met1))

        pre3sum = sum(model.predict(vadata))
        pre3tot = vadata.shape[0]
        met2 = cperf(pre3sum, pre3tot, -1)
        print("False positive rate of the model is      : {}".format(met2))

        # add for performance measurements
        dacc1.append(met1)
        dfpr1.append(met2)


        # iterator counter
        ctr += 1

        # delete data before 'resampling'
        del trdata
        del vadata

    # add to metrics
    acc1.append(np.mean(dacc1))
    fpr1.append(np.mean(dfpr1))

    acc1er.append(np.std(dacc1))
    fpr1er.append(np.std(dfpr1))

    myprint("Cross Validation results after nu= {}:".format(nu1))
    myprint("Average precision is {} +/- {}".format(np.mean(dacc1), np.std(dacc1)))
    myprint("Average fall out  is {} +/- {}".format(np.mean(dfpr1), np.std(dfpr1)))
    




with open(resultsfile, 'a') as ha:
    ha.write('\n')


# save calculation results !
# documentation: nu = np.arange(0.1, 1.01, 0.1)
with open('../data/e12-acc1-1.p', 'wb') as ha:
    pickle.dump(acc1, ha)

with open('../data/e12-fpr1-1.p', 'wb') as ha:
    pickle.dump(fpr1, ha)

with open('../data/e12-acc1er-1.p', 'wb') as ha:
    pickle.dump(acc1er, ha)

with open('../data/e12-fpr1er-1.p', 'wb') as ha:
    pickle.dump(fpr1er, ha)


# random classifier
rand = np.arange(0, 1.01, 0.2)

# This is the ROC curve

#plt.plot(fpr1, acc1, 'r', label='1class svm')
#plt.scatter(fpr1, acc1, c='r', label='1class svm', marker = "D")

plt.errorbar(fpr1, acc1, xerr=fpr1er, yerr=acc1er, fmt='o')
plt.plot(rand, rand, 'k', label='random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("1-class SVM classifier's CV performance")
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
#plt.show() 
plt.savefig('../pictures/e12-1csvm-v1.eps')
plt.close()
