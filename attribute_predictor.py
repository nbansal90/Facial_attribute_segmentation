""" This File contanins the main training code for attribute predictor network"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from dataset_builder import HelenData,UtkData
from network_model_predictor import attribute_predictor

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--ngpu', default=1, type=int,
                    help='total number of gpus (default: 1)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)

def main():
	global args
	args = parser.parse_args()
	kwargs = {'num_workers': 1, 'pin_memory': True}

	"""dataset_train = HelenData('/home/bansa01/taleb/SmithCVPR2013_dataset_resized/','names.txt')
	train_loader = DataLoader(dataset_train, batch_size=1,
		shuffle=True, **kwargs)"""

	dataset_test = UtkData('/home/bansa01/taleb/crop_part1/','/home/bansa01/taleb/', 'label.txt')
	train_loader = DataLoader(dataset_train, batch_size=1,
		shuffle=True, **kwargs)

	model = attribute_predictor()

	# get the number of model parameters
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	model = model.cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))	
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
			momentum=args.momentum, nesterov = args.nesterov,
			weight_decay=args.weight_decay)

		
 
	for epoch in range(args.start_epoch, args.epochs):
		train(train_loader, model, criterion, optimizer,epoch)
		
def train(train_loader, model, criterion, optimizer, epoch):
	"""Train for one epoch on the training set"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.train()

	end = time.time()

	for i, (input, target, img) in enumerate(train_loader):
		target = target.cuda(async=True)
		input = input.float()
		input = input.cuda()
		
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)

		output = model(input_var,img)
		loss = criterion(output, target_var)

		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		batch_time.update(time.time() - end)
		end = time.time()

		print('Epoch: [{0}][{1}/{2}]\t'
			'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1))
"""		
def validate(val_loader, model, criterion, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	
	model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input = input.cuda()
		with torch.no_grad():
			input_var = torch.autograd.Variable(input)
			target_var = torch.autograd.Variable(target)

		output = model(input_var)
		loss = criterion(output, target_var)
		
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		batch_time.update(time.time() - end)
		end = time.time()

	print('Validate * Prec@1 {top1.avg:.3f}'.format(top1=top1))
"""

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count =0

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

if __name__ == '__main__':
	main()

