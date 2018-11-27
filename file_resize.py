""" Resize all the Images and respective Label files to (200,200)"""


import os
import cv2
from skimage import io, transform
import numpy as np

folder_dataset = '/home/bansa01/taleb/SmithCVPR2013_dataset_resized/'

#files = os.listdir(folder_dataset)
#os.chdir(folder_dataset)

image_path  = os.path.join(folder_dataset, 'images')
label_path  = os.path.join(folder_dataset, 'labels')
files = os.listdir(image_path)


for file in files:
	label_image =  (file.split('.')[0])
	img = os.path.join(image_path, file)
	img = cv2.imread(img)
	height,width,_ = (img.shape)
	os.chdir(image_path)

	dim_min = min(height,width)
	dim_max = max(height,width)
	
	
	if (dim_min > 200):
		img_new = cv2.resize(img,(200,200),interpolation=cv2.INTER_AREA)
	elif (dim_max < 200):
		img_new = cv2.resize(img,(200,200),interpolation=cv2.INTER_CUBIC)
	else:
		img_new = cv2.resize(img,(200,200))

	cv2.imwrite(file, img_new)
	
	#Chaning the Label Image
	os.chdir(label_path)
	label_folder = os.path.join(os.getcwd(), label_image)
	label_folder_files = os.listdir(label_folder)
	os.chdir(label_folder)	
	if (dim_min > 200):
		for lff in label_folder_files:
			limg = os.path.join(label_folder, lff)
			limg = cv2.imread(limg)
			limg_new =  cv2.resize(limg,(200,200),interpolation=cv2.INTER_AREA)
			cv2.imwrite(lff, limg_new)
	elif (dim_max < 200):
		for lff in label_folder_files:
			limg = os.path.join(label_folder, lff)
			limg = cv2.imread(limg)
			limg_new = cv2.resize(limg,(200,200),interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(lff, limg_new)
	else:
		for lff in label_folder_files:
			limg = os.path.join(label_folder, lff)
			limg = cv2.imread(limg)
			limg_new = cv2.resize(limg,(200,200))
			cv2.imwrite(lff, limg_new)
			


"""
for file in files:
	label_image =  (file.split('.')[0])
	img = os.path.join(image_path, file)
	img = cv2.imread(img)
	height,width,_ = (img.shape)
	print (height,width)
	
	os.chdir(label_path)
	label_folder = os.path.join(os.getcwd(), label_image)
	for f in os.listdir(label_folder):
		img = os.path.join(label_folder, f)
		img = cv2.imread(img)
		height,width,_ = (img.shape)
		print (height,width)
	break"""
