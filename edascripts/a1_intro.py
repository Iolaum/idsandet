#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files.

from os import listdir
from os.path import isfile, join
#import csv


print("Starting Introduction Script!")

# # Get list of all data files in mypath directory
mypath = "../data/training/"
files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

print("Training files:")
for il in range(10):
	print files[il]

end = len(files)
print("...")


for il in range(end-10, end):
	print files[il]


# print some files:
for it in range(5):

	print("Printing file: {}".format(files[it]))

	datat = []
	data = []
	with open(files[it], 'r') as handle:
		for raw in handle.readlines():
			temp = raw.split()
			datat.append(temp)
	print datat[0]

	for im in datat[0]:
		data.append(int(im))

	print data




# need remove last space

