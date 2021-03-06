#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:27:22 2018
Train script
@author: jakub
"""

from dataset_coco import Dataset
from fcnn_autoenkoder import fcnn 

dataDir = '/home/jakub/data/fxi/coco'

dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                   batchSize=1)
net = fcnn(train = True)
net.train(dataset)
dataset.endDataset()
