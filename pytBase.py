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
from dataset_coco import Dataset
#from torchviz import make_dot

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
        x = x.permute(0,2,1,3,4) # batch, 2, classes, dim, dim)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        return x
        
net = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

dataDir = '/home/jakub/data/coco'
dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                   batchSize=8, minLabelArea=16)

for epoch in range(10):
    runningLoss = 0.0
    i = 0
    for inputs, targets, masks in dataset.iterateMinibatches(val=False):
        # wrap them in Variable
        inputs = Variable(torch.FloatTensor(inputs)) 
        targets = Variable(torch.FloatTensor(targets)) 
        masks = Variable(torch.FloatTensor(masks))
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # print statistics
        runningLoss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, runningLoss / 2000))
            running_loss = 0.0
        i+=1
x = Variable(torch.randn(8,1,256,256))    
y = net.forward(x)
#make_dot(y)