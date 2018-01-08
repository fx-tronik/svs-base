#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:48:33 2018
Convolutional neural network class
@author: jakub
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

from utils import softmax, logSoftmax
from nn_base import nnBase

class fcnn(nnBase):
    numClasses = 29
    networkScale = 8

    imageSize = 256
    targetSize = 24
    batchSize = 24
    
    convFilters0 = 32
    convFilters1 = 64
    convFilters2 = 128
    convFilters3 = 256
    
    def __init__(self, modelWeights=None, train=True):
        nnBase.__init__(self, modelWeights, train=train)
        
    def buildNN(self, modelFile, inputVar, train=True):
        pad = 0 if train else 'same'
        nonlin = lasagne.nonlinearities.rectify
        network = L.InputLayer(shape=(None, 1, None, None),
                input_var=inputVar)
        # Convolutional layers #0
        network = L.Conv2DLayer(network, num_filters=self.convFilters0, 
                                filter_size=(5, 5),nonlinearity=nonlin, 
                                pad=pad)
        network = L.Conv2DLayer(network, num_filters=self.convFilters0, 
                                filter_size=(5, 5),nonlinearity=nonlin, 
                                pad=pad)

        # Max-pooling layer
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layers #1
        network = L.Conv2DLayer(network, num_filters=self.convFilters1, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad)
        network = L.Conv2DLayer(network, num_filters=self.convFilters1, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad)

        # Max-pooling layer
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layers #2
        network = L.Conv2DLayer(network, num_filters=self.convFilters2, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad)
        network = L.Conv2DLayer(network, num_filters=self.convFilters2, 
                        filter_size=(3, 3),nonlinearity=nonlin, 
                        pad=pad)

        # Max-pooling layer
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layers #3
        network = L.Conv2DLayer(network, num_filters=self.convFilters3, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad)
        network = L.Conv2DLayer(network, num_filters=self.convFilters3, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad)

        # Final convolutional layer
        lastNonlin = logSoftmax if train else softmax
        network = L.Conv2DLayer(network, num_filters=2*self.numClasses, 
                                filter_size=(1, 1), nonlinearity=lastNonlin)
        if modelFile:
            modelWeights = np.load(modelFile)
            modelWeights = modelWeights[modelWeights.keys()[0]]
            L.set_all_param_values(network, modelWeights)

        return network
    @staticmethod
    def getNumClasses(self):
        return self.numClasses
    
    @staticmethod
    def getNetworkScale(self):
        return self.networkScale 
       
if __name__ == "__main__":
    net = fcnn(train = True)