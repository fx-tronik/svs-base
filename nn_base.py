#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:51:43 2018
Neural network base class
@author: jakub
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import cv2
from utils import categoricalCrossentropyLogdomain


class nnBase(object):
    def buildNN(self, modelFile, inputVar):
        raise NotImplementedError
        
    @staticmethod
    def getNetworkScale():
        raise NotImplementedError
        
    @staticmethod
    def getReceptiveFieldWidth(self):
        raise NotImplementedError
        
    @staticmethod
    def getReceptiveFieldHeight(self):
        raise NotImplementedError
        
    @staticmethod
    def getNumClasses(self):
        raise NotImplementedError
        
    def __init__(self, modelWeights=None, train=True):
        self.inputVar = T.tensor4('input')
        self.network = self.buildNN(modelWeights, self.inputVar, train=train)
    def compileTrainFunctions(self, learningRate=0.0001):
        # Prepare Theano variables for inputs and targets
        inputVar = self.inputVar
        targetVar = T.tensor4('targets')
        weightVar = T.tensor4('targets')
        
        model = self.network
        prediction = L.get_output(model)
        loss = categoricalCrossentropyLogdomain(prediction, targetVar)
        loss = lasagne.objectives.aggregate(loss, weightVar, mode='mean')
        params = L.get_all_params(model, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=learningRate)
        
        valPrediction =L.get_output(model, deterministic=True)
        valLoss = categoricalCrossentropyLogdomain(valPrediction, targetVar)
        valLoss = lasagne.objectives.aggregate(valLoss, weightVar, mode='mean')
        valAcc = T.sum(T.eq(T.argmax(valPrediction, axis=1), \
                            T.argmax(targetVar, axis=1))*weightVar, 
                      dtype=theano.config.floatX)
        valAcc /= T.sum(weightVar)  
        
        trainFn = theano.function([inputVar, targetVar, weightVar], loss, 
                                  updates=updates, allow_input_downcast=True)
        valFn = theano.function([inputVar, targetVar, weightVar], 
                                 [valLoss, valAcc], allow_input_downcast=True)
        self.trainFn = trainFn
        self.valFn = valFn
        
    def compileTestFunctions(self):
        inputVar = self.inputVar
        model = self.network
        prediction = L.get_output(model, deterministic=True)
        forwardFn = theano.function([inputVar], prediction, 
                                    allow_input_downcast=True)
        self.forwardFn = forwardFn
        
        
        