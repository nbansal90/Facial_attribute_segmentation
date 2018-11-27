import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

path = "/home/bansa01/taleb/image_output.txt"

class attribute_predictor(nn.Module):

        def __init__(self):
                super(attribute_predictor,self).__init__()

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
                self.conv51 = nn.Conv2d(in_channels=512,out_channels=1024, kernel_size=3,stride=1, padding=1)
                self.conv52 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3,stride=1, padding=1)
                self.bn5 = nn.BatchNorm2d(1024)

                #Defining Relu and Maxpool Layer
                self.prelu = nn.PReLU()
                self.maxpool= nn.MaxPool2d(kernel_size=2,stride=2)
		self.avgpool = nn.AvgPool2d((12,12))
		self.fc = nn.Linear(1024, 2)
		self.sm = nn.Softmax2d()
                #Inittializing the module
                for m in self.modules():
                        if isinstance(m, nn.Conv2d):
                                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                                if m.bias is not None:
                                        nn.init.constant_(m.bias,0)
                        elif isinstance(m, nn.BatchNorm2d):
                                nn.init.constant_(m.weight, 1)
                                nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.Linear):
                                nn.init.normal_(m.weight, 0, 0.01)
                                nn.init.constant_(m.bias, 0)

	def forward(self, x, y):
                
                #Block 1
                out = self.prelu(self.bn1(self.conv11(x)))
                out = self.prelu(self.bn1(self.conv12(out)))
                out = self.maxpool(out)

                #Block 2
                out = self.prelu(self.bn2(self.conv21(out)))
                out = self.prelu(self.bn2(self.conv22(out)))
                out = self.maxpool(out)

                #Block 3
                out = self.prelu(self.bn3(self.conv31(out)))
                out = self.prelu(self.bn3(self.conv32(out)))
                out = self.maxpool(out)

                #Block 4
                out = self.prelu(self.bn4(self.conv41(out)))
                out = self.prelu(self.bn4(self.conv42(out)))
                out = self.maxpool(out)
                #Block of 512
                out = self.prelu(self.bn4(self.conv42(out)))
                out = self.prelu(self.bn4(self.conv42(out)))
                #Block of 1024
                out = self.prelu(self.bn5(self.conv51(out)))
                out = self.prelu(self.bn5(self.conv52(out)))
		
		#Adding the Segmentation Part Contribution
		output_dict = pickle.load( open( path, "rb" ) )
		output_tensor = output_dict[y]
		
		out = broad_multiply(out, output_tensor)		
		out = self.avgpool(out).view(7,1024)
		
		out2 = self.fc(out)
		out3 = self.sm(out)
		
		out = torch.mul(out2, out3)
		out = torch.sum(out, 0)	
                return out

def broad_multiply(model_output, semantic_region):
	""" model_output = 1024 x 12 x 12  semantic_region  = 7 x 12 x 12"""
	a,_,_= semantic_region.shape
	res = Variable(torch.zeros(1024,12,12))
	ll = []

	for i in range(0,a):
		tmp = semantic_region[i][:][:]
		b,c = tmp.shape
		for j in range(0,b):
			for k in range(0,c):
				res = res + model_output * (tmp[j][k].expand_as(model_output))
		ll.append(res)
		res = Variable(torch.zeros(1024,12,12))
	c = torch.stack((ll))

	reutrn c
	
