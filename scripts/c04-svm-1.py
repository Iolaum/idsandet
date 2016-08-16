#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement an SVM classifier to create ROC curves.


import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# where to log results
resultsfile = '../data/c04-svm-1-results.txt'
datafile = '../data/c04-svm-1-'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

myprint("Starting a linear SVM classifier.")

myprint("Loading training data.")
trdat = np.load('../data/b5_trmatrix.npy')


# Adduser attack - 10
# hydra ftp - 0.05
# hydra ssh - 0.5
# java meter - 0.1
# meterpreter - 10
# webshell - 1

# define SVM parameters
# cpar 0.05 - 0.1 - 0.5 - 1 - 5 - 10 - 50
#
# optimum regularisation parameter (C) for each attack:
# 1:10 - 2:0.05 - 3:0.5 - 4:0.1 - 5:10 - 6:1
cpar = 1

# attack data options
atdatselector = 6

if atdatselector == 1:
    myprint("Loading adduser attack data.")
    atdat = np.load('../data/b5_at1mat_adduser.npy')

elif atdatselector == 2:
    myprint("Loading hydra ftp attack data.")
    atdat = np.load('../data/b5_at2mat_hyftp.npy')

elif atdatselector == 3:
    myprint("Loading hydra ssh attack data.")
    atdat = np.load('../data/b5_at3mat_hyssh.npy')

elif atdatselector == 4:
    myprint("Loading java meterpreter attack data.")
    atdat = np.load('../data/b5_at4mat_javamet.npy')

elif atdatselector == 5:
    myprint("Loading meterpreter attack data.")
    atdat = np.load('../data/b5_at5mat_meter.npy')

elif atdatselector == 6:
    myprint("Loading web shell attack data.")
    atdat = np.load('../data/b5_at6mat_webshell.npy')


myprint("Loading validation data.")
vadat = np.load('../data/b5_vamatrix.npy')

# split attach set
at1ctr = atdat.shape[0]/2


# use to train
at1 = atdat[0:at1ctr, :]

# use for accuracy
at2 = atdat[at1ctr:, :]

# create classification labels
l1 = np.zeros(trdat.shape[0])
l2 = np.ones(at1.shape[0])


# convatenate to create training set
x = np.concatenate((trdat, at1), axis=0)
y = np.concatenate((l1, l2), axis=0)

# debug
# print x.shape
# print y.shape

# initialise SVC class
model = SVC(C=cpar, kernel='linear', max_iter=5000, verbose=False, class_weight='balanced', probability=True)

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

# Probability = True
# http://stackoverflow.com/a/15019805/1904901

myprint("Starting training an SVM model with linear kernel and C={}.".format(cpar))

# model.fit(x[830:930, :], y[830:930])
model.fit(x, y)

# log support vectors
myprint('Number of Support Vectors for normal class: {}'.format(model.n_support_[0]))
myprint('Number of Support Vectors for attack class: {}'.format(model.n_support_[1]))

pre1 = model.predict_proba(at2)
# # debug
# print pre1
# exit()
# [prob_normal, prob_attack]

pre1a1 = float(at2.shape[0])

# decision thresshold (attack)
prthrng = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.]

# create performance objects
svmacc = []
svmfpr = []

# calculate stats for various decision thressholds
for prth in prthrng:

    myprint("Decision thresshold for attack classification is {}".format(prth))

    #calculate attack detection accuracy
    pre1a2 = 0

    for it1 in pre1[:, 1]:
        if it1 >= prth:
            pre1a2 += 1
     
    attacc = pre1a2/pre1a1
    svmacc.append(attacc)
    myprint("Attack detection accuracy in our SVM model is            {}".format(attacc))

    pre2 = model.predict_proba(vadat)
    pre2a1 = float(vadat.shape[0])

    #calculate attack detection accuracy
    pre2a2 = 0

    for it1 in pre2[:, 1]:
        if it1 >= prth:
            pre2a2 += 1

    attfpr = pre2a2/pre2a1
    svmfpr.append(attfpr)
    myprint("Attack detection false positive rate in our SVM model is {}".format(attfpr))


with open(resultsfile, 'a') as ha:
    ha.write('\n')

savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'wb') as ha:
    pickle.dump(svmacc, ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'wb') as ha:
    pickle.dump(svmfpr, ha)
