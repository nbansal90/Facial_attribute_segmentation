#Intermediatoy file just for renaming the Image files

import os 
import cv2
from skimage import io, transform
import numpy as np

folder_dataset = '/home/bansa01/taleb/crop_part1/'

files = os.listdir(folder_dataset)
os.chdir(folder_dataset)

for f in files:
	a = f.split('.')
	b = ".".join(a[:2])
	os.rename(f,b)

