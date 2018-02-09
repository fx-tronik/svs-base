#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:43:03 2018
Klasa bazowa dla sieci w PyTorch
@author: jakub
"""
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot

class Net(nn.Module):
    networkName = 'fullyConvolutional_v00_MPII'

    numClasses = 17
    networkScale = 8

    imageSize = 256
    targetSize = 24
    batchSize = 24
    
    convFilters1 = 32
    convFilters2 = 64
    convFilters3 = 128
    convFilters4 = 256
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv11 = nn.Conv2d(1,self.convFilters1, (5,5))
        self.conv12 = nn.Conv2d(self.convFilters1,self.convFilters1, (5,5))
        self.pool = nn.MaxPool2d((2,2))
        
        self.conv21 = nn.Conv2d(self.convFilters1,self.convFilters2, (3,3))
        self.conv22 = nn.Conv2d(self.convFilters2,self.convFilters2, (3,3))
        
        self.conv31 = nn.Conv2d(self.convFilters2,self.convFilters3, (3,3))
        self.conv32 = nn.Conv2d(self.convFilters3,self.convFilters3, (3,3))
        
        self.conv41 = nn.Conv2d(self.convFilters3,self.convFilters4, (3,3))
        self.conv42 = nn.Conv2d(self.convFilters4,self.convFilters4, (3,3))
        
        self.conv5 = nn.Conv2d(self.convFilters4, 2*self.numClasses, (1,1))
    def forward(self, x):
        x = self.pool(F.relu(self.conv12(F.relu(self.conv11(x)))))
        x = self.pool(F.relu(self.conv22(F.relu(self.conv21(x)))))
        x = self.pool(F.relu(self.conv32(F.relu(self.conv31(x)))))
        x = self.pool(F.relu(self.conv42(F.relu(self.conv41(x)))))
        x = self.conv5(x)
        xShape = x.size()
        x = x.view(-1, self.numClasses, 2, xShape[-2], xShape[-1])
        x = F.log_softmax(x, dim = 2)
        return x
        
net = Net()
criterion = nn.NLLLoss()


x = Variable(torch.randn(8,1,256,256))    
y = net.forward(x)
make_dot(y)