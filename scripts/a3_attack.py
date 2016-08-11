#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all attack data files and create an
# attack data dictionary of lists.

# Put all the attack set data in /data/attack/

from __future__ import division
from os import listdir, stat, walk
from os.path import isfile, join
import pickle


print("Starting Preprocessing attack data!")

# # Get list of all data directories [dirs]

mypath = "../data/attack/"
dirs = []
for (dirpath, dirnames, filenames) in walk(mypath):
    dirs.extend(dirnames)
    break
dirs.sort()

# debug
# print dirs

for key, value in enumerate(dirs):
	dirs[key] = join(mypath, value)

# debug
print("There are {} attack data directories.".format(len(dirs)))


# get list of all files in directories
filelist = []
for il in dirs:
    tfilelist = []

    for (dirpath, dirnames, filenames) in walk(il):
        tfilelist.extend(filenames)
        tfilelist.sort()
        # dfile = "prp3_" + dirpath.split("/")[-1] + ".txt"
        break
    for key, value in enumerate(tfilelist):
        tfilelist[key] = join(il, value)

    filelist.extend(tfilelist)

# debug
# print filelist
print("There are {} attack data files.".format(len(filelist)))
# print filelist[12]

# find part to keep as name
# print(filelist[12].index('U'))
# print(filelist[12].index('UAD'))
# print(filelist[12][filelist[12].index('UAD'):-4])

print("Starting creating attack data dictionary")


atdata = {}

# iterate through all files
for it in filelist:
    datat = []
    data = []
    with open(it, 'r') as handle:
        # make each line into a list
        # data have only one line - otherwise the next step wouldn't work
        for raw in handle.readlines():
            temp = raw.split()
            datat.extend(temp)

    for im in datat:
        data.append(int(im))

    # debug
    # print data
    # import sys
    # sys.exit("Error message")

    atdata[it[it.index('UAD'):-4]] = data

print("Saving attack data dictionary!")

with open('../data/1_attack.p', 'wb') as handle:
    pickle.dump(atdata, handle)

# check file size
statinfo = stat('../data/1_attack.p')

print("Attack data dictionary saved. It's size is {} kbytes".
      format(int(statinfo.st_size/1024)))

