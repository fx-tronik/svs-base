#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 08:11:09 2018
Keras Base
@author: jakub
"""
import time
from keras.models import Sequential
from keras.layers import Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from dataset_coco import Dataset
from theanoFunctions import logSoftmax2, categoricalCrossentropyLogdomain2

numClasses = 29
networkScale = 8

imageSize = 256
targetSize = 24
batchSize = 3

convFilters0 = 32
convFilters1 = 64
convFilters2 = 128
convFilters3 = 256

numEpochs = 10

inputShape = (1, imageSize, imageSize)

model = Sequential()
model.add(Conv2D(filters=convFilters0, kernel_size=(5,5),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters0, kernel_size=(5,5),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilters1, kernel_size=(3,3),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters1, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilters2, kernel_size=(3,3),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters2, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilters3, kernel_size=(3,3),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters3, kernel_size=(3,3),
                 activation='relu'))

model.add(Conv2D(filters=2*numClasses, kernel_size=(1,1),
                 activation='linear'))
model.add(Reshape((numClasses, 2, targetSize, targetSize)))
model.add(Activation(logSoftmax2))

model.compile(loss=categoricalCrossentropyLogdomain2,
              optimizer=Adam(lr=0.0004))#,
              #metrics=['accuracy'])

dataDir = '/home/jakub/data/coco'

dataset  = Dataset(dataDir=dataDir, imageSize=imageSize, targetSize=targetSize, 
                   batchSize=1)
for epoch in range(numEpochs):
    trainErr = 0
    trainBatches = 0
    startTime = time.time()
    
    for batch in dataset.iterateMinibatches():
        inputs, targets, weights = batch
        trainErr += model.train_on_batch(inputs, targets)
        trainBatches += 1
    valErr = 0
    valBatches = 0
    valAcc = 0
    for batch in dataset.iterateMinibatches(True):
        inputs, targets, weights = batch
        valErr += model.test_on_batch(inputs, targets)
        valBatches += 1
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, numEpochs, time.time() - startTime))
    print("  training loss:\t\t{:.6f}".format(trainErr / trainBatches))
    print("  validation loss:\t\t{:.6f}".format(valErr / valBatches))
