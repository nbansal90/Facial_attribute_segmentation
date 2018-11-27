#File takens in Helen Dataset and changes the number of Labels from 11 to 7
# incaccordance with the setup mentioned in the paper

import numpy as np
import cv2
import os
from os import listdir
import re
import sys

#Chaning the Directory as the Label Directory
os.chdir('/home/bansa01/taleb/SmithCVPR2013_dataset_resized/labels/')
path = (os.getcwd())

files = listdir(path)
#print (len(files))

#Merging the two eye brows
def merge_eyebrow(img1, img2, image_folder, label_folder):
	tmp_img1 = img1
	tmp_img2 = img2
	pref_str = re.findall(r'(.*)lbl.*', img1)
	if (len(pref_str) == 0):
		print ("Not able to fetch the Image Name Prefix")
		sys.exit()
	pref_str = str(pref_str[0])
	pref_str = pref_str + 'lbl02.png'

	img1 = label_folder + '/' + image_folder + '/' + img1
	img2 = label_folder + '/' + image_folder + '/' + img2
	curr_path = label_folder + '/' + image_folder
	os.chdir(curr_path)

	img1 = cv2.imread(img1)
	img2 = cv2.imread(img2)

	arr1 = np.asarray(img1)
	arr2 = np.asarray(img2)
	arr1 = np.transpose(arr1, (2,0,1))
	arr2 = np.transpose(arr2,(2,0,1))

	(c,h,w) = (arr2.shape)
	arr_new = np.zeros((c,h,w))

	#Replacing the value at index (i,j) with the maximum of two
	for i in range(0,c):
		for j in range(0,h):
			for k in range(0,w):
				arr_new[i][j][k] = max([arr1[i][j][k], arr2[i][j][k]])
	arr_new =  np.transpose(arr_new, (1,2,0))
	os.remove(tmp_img1)
	os.remove(tmp_img2)
	cv2.imwrite(pref_str, arr_new)
		
#Merging two Eyes
def merge_eyes(img1, img2,image_folder,label_folder):
	tmp_img1 = img1
	tmp_img2 = img2
	pref_str = re.findall(r'(.*)lbl.*', img1)
	if (len(pref_str) == 0):
		print ("Not able to fetch the Image Name Prefix")
		sys.exit()
	pref_str = str(pref_str[0])
	pref_str = pref_str + 'lbl03.png'

	img1 = label_folder + '/' + image_folder + '/' + img1
	img2 = label_folder + '/' + image_folder + '/' + img2
	curr_path = label_folder + '/' + image_folder
	os.chdir(curr_path)

	img1 = cv2.imread(img1)
	img2 = cv2.imread(img2)

	arr1 = np.asarray(img1)
	arr2 = np.asarray(img2)
	arr1 = np.transpose(arr1, (2,0,1))
	arr2 = np.transpose(arr2,(2,0,1))

	(c,h,w) = (arr2.shape)
	arr_new = np.zeros((c,h,w))

	#Replacing the value at index (i,j) with the maximum of two
	for i in range(0,c):
		for j in range(0,h):
			for k in range(0,w):
				arr_new[i][j][k] = max([arr1[i][j][k], arr2[i][j][k]])
	arr_new =  np.transpose(arr_new, (1,2,0))
	os.remove(tmp_img1)
	os.remove(tmp_img2)
	cv2.imwrite(pref_str, arr_new)

#Mergeing the Mouth and Lips
def merge_mouth(img1, img2, img3,image_folder,label_folder):
	tmp_img1 = img1
	tmp_img2 = img2
	tmp_img3 = img3

	pref_str = re.findall(r'(.*)lbl.*', img1)
	if (len(pref_str) == 0):
		print ("Not able to fetch the Image Name Prefix")
		sys.exit()
	pref_str = str(pref_str[0])
	new_str = pref_str + 'lbl05.png'

	img1 = label_folder + '/' + image_folder + '/' + img1
	img2 = label_folder + '/' + image_folder + '/' + img2
	img3 = label_folder + '/' + image_folder + '/' + img3

	curr_path = label_folder + '/' + image_folder
	os.chdir(curr_path)

	img1 = cv2.imread(img1)
	img2 = cv2.imread(img2)
	img3 = cv2.imread(img3)

	arr1 = np.asarray(img1)
	arr2 = np.asarray(img2)
	arr3 = np.asarray(img3)

	arr1 = np.transpose(arr1, (2,0,1))
	arr2 = np.transpose(arr2,(2,0,1))
	arr3 = np.transpose(arr3,(2,0,1))

	(c,h,w) = (arr2.shape)
	arr_new = np.zeros((c,h,w))

	#Replacing the value at index (i,j) with the maximum of two
	for i in range(0,c):
		for j in range(0,h):
			for k in range(0,w):
				arr_new[i][j][k] = max([arr1[i][j][k], arr2[i][j][k], arr3[i][j][k]])
	arr_new =  np.transpose(arr_new, (1,2,0))
	os.remove(tmp_img1)
	os.remove(tmp_img2)
	os.remove(tmp_img3)
	cv2.imwrite(new_str, arr_new)
	#Making Necessary Changes in terms of Name
	prev_name = pref_str + 'lbl06.png'
	curr_name = pref_str + 'lbl04.png'
	os.rename(prev_name, curr_name)
	curr_name = pref_str + 'lbl06.png'
	prev_name = pref_str + 'lbl10.png'
	os.rename(prev_name, curr_name)

#Reducing the Semantic regions from 11 to 7
def merge_images_seven_group(merge_image_list, image_folder,  label_folder):
	#merge_eyebrow(merge_image_list[0], merge_image_list[1], image_folder, label_folder)
	#merge_eyes(merge_image_list[2], merge_image_list[3],image_folder,label_folder)
	#merge_mouth(merge_image_list[4], merge_image_list[5],merge_image_list[6],image_folder,label_folder)
	pass	

#Convert in to Grey Scale
def  convert_to_greyscale(path, files):
	for f in files:
		p = path + '/' + f
		os.chdir(p)
		images = listdir(p)
		for image in images:
			img = cv2.imread(image)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			cv2.imwrite(image, img)


for f in files:
	p = path + '/' + f
	#print (p)
	os.chdir(p)
	images =  (listdir(p))
	merge_image_list = []
	for image in images:
		#i11 = cv2.imread(image)
		#print (i11.shape)
		r2 = re.findall(r'(.*lbl02.png)', image)
		if (len(r2) > 0):
			merge_image_list.append(str(r2[0]))
		r3 = re.findall(r'(.*lbl03.png)', image)
		if (len(r3) > 0):
                	merge_image_list.append(str(r3[0]))
		r4 = re.findall(r'(.*lbl04.png)', image)
		if (len(r4) > 0):
                	merge_image_list.append(str(r4[0]))
		r5 = re.findall(r'(.*lbl05.png)', image)
		if (len(r5) > 0):
                	merge_image_list.append(str(r5[0]))
		r7 = re.findall(r'(.*lbl07.png)', image)
		if (len(r7) > 0):
                	merge_image_list.append(str(r7[0]))
		r8 = re.findall(r'(.*lbl08.png)', image)
		if (len(r8) > 0):
                	merge_image_list.append(str(r8[0]))
		r9 = re.findall(r'(.*lbl09.png)', image)
		if (len(r9) > 0):
                	merge_image_list.append(str(r9[0]))
	#os.remove(p + '/' + '173153923_2_lbl10.png')
	merge_image_list = sorted(merge_image_list)
	merge_images_seven_group(merge_image_list, f, path)

convert_to_greyscale(path, files)
