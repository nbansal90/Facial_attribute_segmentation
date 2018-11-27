#Dumping the Label of UTK Dataset using pickle, which will be later Used 
#while Building the dataset

import os
import cv2
import numpy as np
import pickle
from collections import defaultdict

folder_dataset = '/home/bansa01/taleb/crop_part1/'

files = os.listdir(folder_dataset)
os.chdir(folder_dataset)

print (os.getcwd())

l = open("/home/bansa01/taleb/label.txt","wb")
labels_dict = defaultdict(list)

for f in files:
	names = f.split('_')
	#['14', '0', '1', '20170104012054585.jpg']
	#Checking GENDER
	if (names[1] == '0'):
		labels_dict[f].append(1)
	else:
		labels_dict[f].append(0)
	#Checking if ASIAN
	if (names[2] == '2'):
		labels_dict[f].append(1)
	else:
		labels_dict[f].append(0)


pickle.dump(labels_dict,l)	
"""
import torch
from torch.autograd import Variable

input = Variable(torch.randn(1, 3, 200, 200))
input1 = Variable(torch.randn(1, 3, 200, 200))
input2 = Variable(torch.randn(1, 3, 200, 200))

labels_dict = defaultdict(list)

labels_dict['a'].append(input)
labels_dict['d'].append(input1)
labels_dict['c'].append(input2)

print (labels_dict)
"""
