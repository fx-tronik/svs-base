#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:27:22 2018
Train script
@author: jakub
"""

#from dataset_coco import Dataset
from datasetMPII import Dataset
from fcnn import fcnn 

dataDir = '/home/jakub/data/MPII'

dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                   batchSize=8)
net = fcnn(train = True)
net.train(dataset)
dataset.endDataset()
