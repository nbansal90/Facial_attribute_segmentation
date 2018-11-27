import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

""" Out shape as part for an Image of size 200 x 200
Encoder

torch.Size([1, 3, 200, 200])
torch.Size([1, 64, 200, 200])
torch.Size([1, 64, 200, 200])
torch.Size([1, 64, 100, 100])
torch.Size([1, 128, 100, 100])
torch.Size([1, 128, 100, 100])
torch.Size([1, 128, 50, 50])
torch.Size([1, 256, 50, 50])
torch.Size([1, 256, 50, 50])
torch.Size([1, 256, 25, 25])
torch.Size([1, 512, 25, 25])
torch.Size([1, 512, 25, 25])
DECODER


torch.Size([1, 512, 25, 25])
torch.Size([1, 512, 25, 25])
torch.Size([1, 256, 25, 25])
torch.Size([1, 256, 50, 50])
torch.Size([1, 256, 50, 50])
torch.Size([1, 128, 50, 50])
torch.Size([1, 128, 100, 100])
torch.Size([1, 128, 100, 100])
torch.Size([1, 64, 100, 100])
torch.Size([1, 64, 200, 200])
torch.Size([1, 64, 200, 200])
torch.Size([1, 7, 200, 200])

at last torch.Size([1, 7, 12, 12])
"""

class encoder_decoder(nn.Module):
	
	def __init__(self):
		super(encoder_decoder,self).__init__()

		#Defining the Encoder Network
		self.conv11 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3,stride=1, padding=1)
		self.conv12 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,stride=1, padding=1)
		self.bn1 =  nn.BatchNorm2d(64)
		self.conv21 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,stride=1, padding=1)
		self.conv22 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1, padding=1)
		self.bn2 =  nn.BatchNorm2d(128)
		self.conv31 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3,stride=1, padding=1)
		self.conv32 = nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3,stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv41 = nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3,stride=1, padding=1)
		self.conv42 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3,stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(512)
		
		#Defining Relu and Maxpool Layer
		self.prelu = nn.PReLU()
		self.maxpool= nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

		#Defining Decoder Network
		self.deconv41=nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1, padding=1)
		self.deconv31=nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=1, padding=1)
		self.deconv32=nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,stride=1, padding=1)
		self.deconv21=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=1, padding=1)
		self.deconv22=nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=1, padding=1)
		self.deconv11=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=1, padding=1)
		self.deconv12=nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1, padding=1)
		self.deconv13=nn.ConvTranspose2d(in_channels=64,out_channels=7,kernel_size=3,stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(7)
		#Defining UnMaxpool Layer
		self.maxunpool=nn.MaxUnpool2d(kernel_size=2, stride = 2)
		
		#Defining Last Pooling Layers
		self.maxpool1= nn.MaxPool2d(kernel_size=2,stride=2)
		self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)

		#Inittializing the module
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal(m.weight, mode='fan_out')
				if m.bias is not None:	
					nn.init.constant(m.bias,0)
			elif isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.constant(m.bias,0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant(m.weight, 1)
				nn.init.constant(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal(m.weight, 0, 0.01)
				nn.init.constant(m.bias, 0)
	
	def forward(self, x):
		
		#Block 1
		out = self.prelu(self.bn1(self.conv11(x)))
		out = self.prelu(self.bn1(self.conv12(out)))
		size1 = out.size()
		out,ind1 = self.maxpool(out)

		#Block 2
		out = self.prelu(self.bn2(self.conv21(out)))
		out = self.prelu(self.bn2(self.conv22(out)))
		size2 = out.size()
		out,ind2 = self.maxpool(out)

		#Block 3
		out = self.prelu(self.bn3(self.conv31(out)))
		out = self.prelu(self.bn3(self.conv32(out)))
		size3 = out.size()
		out,ind3 = self.maxpool(out)

		#Block 4
		out = self.prelu(self.bn4(self.conv41(out)))
		out = self.prelu(self.bn4(self.conv42(out)))

		#Block 4
		out = self.prelu(self.bn4(self.deconv41(out)))
		out = self.prelu(self.bn4(self.deconv41(out)))

		#Block 3
		out = self.prelu(self.bn3(self.deconv31(out)))
		out  = self.maxunpool(out,ind3,size3)
		out = self.prelu(self.bn3(self.deconv32(out)))

		#Blcok 2
		out = self.prelu(self.bn2(self.deconv21(out)))
		out  = self.maxunpool(out,ind2,size2)
		out = self.prelu(self.bn2(self.deconv22(out)))

		#Block 1
		out = self.prelu(self.bn1(self.deconv11(out)))
		out  = self.maxunpool(out,ind1,size1)
		out = self.prelu(self.bn1(self.deconv12(out)))
		out = self.prelu(self.bn5(self.deconv13(out)))
		
		#Last Pooling layers
		out = self.maxpool1(input2)
		out = self.avgpool(out)
		out = self.avgpool(out)
		out = self.maxpool1(out)
	
		return out
