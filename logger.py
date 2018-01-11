#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:35:14 2018
Logger 
@author: jakub
"""
import os
from time import time
from datetime import datetime

def currentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

class Logger:
    def __init__(self):
        plotsPath = os.path.abspath('./plots')
        logsPath = os.path.abspath('./logs')
        mainlogPath = os.path.join(logsPath, 'main.log')
        
        self.paramsToTrack(['trainLoss', 'valLoss', 'valAcc'])
        
        self.plotsPath = plotsPath
        self.logsPath = logsPath
        self.mainlogPath = mainlogPath
        
    def paramsToTrack(self, params = []):
        self.container = None
        self.parNames = params
        self.container = {parName:[] for parName in params}
        
    
        
    def procesEpoch(self, *params):
        container = self.container
        if len(params) != len(container):
            raise ValueError('All params should be initialized!')
        for val, key in zip(params, self.parNames):
            container[key].append(val)
        
        