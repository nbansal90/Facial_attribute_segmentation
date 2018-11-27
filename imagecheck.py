# Checking some stuff, not part of cord flow

import numpy as np
import cv2
import os
from os import listdir
import re
import sys
import random

"""
#Chaning the Directory as the Label Directory
os.chdir('/home/bansa01/taleb/SmithCVPR2013_dataset_resized/images/')
print (os.getcwd())

img1 = '2933512856_1.jpg'
img2 = '2962437091_1.jpg'

i1 = cv2.imread(img1)
i2 = cv2.imread(img2)

print (i1.shape)
print (i2.shape)
"""

os.chdir('/home/bansa01/taleb/crop_part1/')
print (os.getcwd())

files = listdir(os.getcwd())
random.shuffle(files)

count = 0
for f in files:
	i = cv2.imread(f)
	print (i.shape)
	count = count + 1
	if count > 20:
		break


os.chdir('/home/bansa01/taleb/UTKFace/')
print (os.getcwd())

files = listdir(os.getcwd())
random.shuffle(files)
count = 0

for f in files:
	i = cv2.imread(f)
	print (i.shape)
	count = count + 1
	if count > 20:
		break
