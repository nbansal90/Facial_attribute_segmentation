""" Dataset Builder File"""

from __future__ import print_function, division

import torch
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
from skimage import io, transform
from torchvision import transforms

#folder_dataset = /home/bansa01/taleb/SmithCVPR2013_dataset_resized/

#Building Helen Dataset
class HelenData(Dataset):

	def __init__(self, folder_dataset, training_file, transform=None):
		super(HelenData,self).__init__()
		self.transform  = transform
		self.file_name = []
		self.images = 'images'
		self.labels = 'labels'
		self.root = folder_dataset

		with open(folder_dataset + training_file) as f:
			for line in f:
				self.file_name.append(line.rstrip())
		
	def __getitem__(self, index):
		image_folder = os.path.join(self.root, self.images)
		img_name = os.path.join(image_folder, self.file_name[index]+ '.jpg')
		img_ret = self.file_name[index] + '.jpg'

		img = io.imread(img_name)
		img = np.transpose(img, (2,0,1))
		
		label_folder = os.path.join(self.root, self.labels)
		label_name =  os.path.join(label_folder, self.file_name[index])
		label_files = os.listdir(label_name)
		
		label_img = []
		labels_list = []
		for label_file in label_files:
			label_img.append(os.path.join(label_name, label_file))
		label_img = sorted(label_img)
		
		for i in label_img:
			tmp = io.imread(i)
			tmp = np.asarray(tmp)
			labels_list.append(tmp)

		label = np.stack(labels_list, axis =0)
		label = torch.from_numpy(label)
		
		"""if self.transform is not None:
			img = Image.fromarray(img)
			img = self.transform(img)"""
		
		#label = torch.from_numpy(label)
		#img = np.asarray(img)
		img = torch.from_numpy(img)
		return (img, label)
		 

	def __len__(self):
		return len (self.file_name)



#Building UTK Dataset
#folder dataset: /home/bansa01/taleb/crop_part1/
#training file: label.txt
#root folder:  /home/bansa01/taleb/
class UtkData(Dataset):
	def __init__(self, folder_dataset, root_folder, training_file, transform=None):
		super(UtkData,self).__init__()
		self.transform = transform
		self.dataset_dir = folder_dataset
		self.data = []
		self.label = []
		
		pickle_path = os.path.join(root_folder + training_file)
		label_dict = pickle.load( open( pickle_path, "rb" ) )
		
		for key, value in label_dict.items():
			(self.data).append(key)
			(self.label).append(value)
		
	def __getitem__(self, index):
		image_loc = os.path.join(self.dataset_dir, self.data[index])
		label_val = self.label[index]
		label_val = np.asarray(label_val)	
		img = io.imread(image_loc)
		img = np.transpose(img, (2,0,1))
		"""if self.transform is not None:
			img = Image.fromarray(img)
			img = self.transform(img)"""
		#img = np.asarray(img)
		img = torch.from_numpy(img)
		label = torch.from_numpy(label_val)
		return (img, label,self.data[index])	

	def __len__(self):
		return len(self.data)

"""
transformed_dataset = HelenData('/home/bansa01/taleb/SmithCVPR2013_dataset_resized/','names.txt')
dataloader = DataLoader(transformed_dataset, batch_size=32,
                        shuffle=True, num_workers=1)
for i, (data, label) in enumerate(dataloader):
	print (data.shape)
	print (label.shape)"""
"""
transformed_dataset = UtkData('/home/bansa01/taleb/crop_part1/','/home/bansa01/taleb/', 'label.txt')
dataloader = DataLoader(transformed_dataset, batch_size=32,
			shuffle=True, num_workers=1)

for i, (data, label, img) in enumerate(dataloader):
	print (data.shape)
	print (label.shape)
	print (img)"""
