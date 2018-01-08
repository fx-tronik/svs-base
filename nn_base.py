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
import cv2


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