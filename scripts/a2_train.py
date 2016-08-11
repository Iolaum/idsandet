#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a 
# training data dictionary of lists.

# Put all the training set data in /data/training

from __future__ import division
from os import listdir, stat
from os.path import isfile, join
import pickle


print("Starting Preprocessing training data!")

# # Get list of all data files in mypath directory
mypath = "../data/training/"
files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


print("Starting creating training data dictionary")

# get only file name for dict type
lf1 = len(mypath)

# debug/test
# print(files[0][lf1:-4])


trdata = {}

# iterate through all files
for it in files:
	datat = []
	data = []
	with open(it, 'r') as handle:
		# make each line into a list
		# data have only one line - otherwise the next step wouldn't work
		for raw in handle.readlines():
			temp = raw.split()
			datat.append(temp)

	for im in datat[0]:
		data.append(int(im))

	# print data

	trdata[it[lf1:-4]]=data

print("Saving training data dictionary!")

with open('../data/1_training.p', 'wb') as handle:
  pickle.dump(trdata, handle)

# check file size
statinfo = stat('../data/1_training.p')

print("Training data dictionary saved. It's size is {} kbytes".format(int(statinfo.st_size/1024)))
