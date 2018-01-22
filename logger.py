#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:35:14 2018
Logger 
@author: jakub
"""
import os
import matplotlib
matplotlib.use('agg', force=True)
plt = matplotlib.pyplot
import hashlib
import time
from datetime import datetime
import numpy as np


def currentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

class Logger:
    def __init__(self, networkName='neuralNetwork'):
        plotPath = os.path.abspath('./plots')
        plotPath = os.path.join(plotPath, '{}.png'.format(networkName))
        logsPath = os.path.abspath('./logs')
        logPath = os.path.join(logsPath, '{}.log'.format(networkName))
        mainlogPath = os.path.join(logsPath, 'main.log')
        hashGen = hashlib.sha1()
        hashGen.update(str(time.time()))
        expId = hashGen.hexdigest()[:10]
        
        self.paramsToTrack(['trainLoss', 'valLoss', 'valAcc'])
        self.plotPath = plotPath
        self.logsPath = logsPath
        self.logPath = logPath
        self.mainlogPath = mainlogPath
        self.networkName = networkName
        self.bestVal = 100.0
        self.epochs = 0
        self.startDate = currentTime()
        self.epochStartTime = time.time()
        self.expId = expId
    def paramsToTrack(self, params = []):
        self.container = None
        self.parNames = params
        self.container = {parName:[] for parName in params}
        
    def processEpoch(self, *params):
        container = self.container
        if len(params) != len(container):
            raise ValueError('All params should be initialized!')
        for val, key in zip(params, self.parNames):
            container[key].append(val)
        self.printVals()
        self.logEpoch()
        self.epochs+=1
        self.epochStartTime = time.time()
    def printVals(self):
        plt.figure()
        for key in self.container:
            epochs = len(self.container[key])
            x = np.arange(epochs)
            y = self.container[key]
            plt.plot(x,y, label=key)
            plt.legend()
        plt.savefig(os.path.join(self.plotPath))
    def logEpoch(self):
        logPath = self.logPath
        with open(logPath, 'a') as f:
            f.write('Epoch {} took {:.3f}s \n'.format(self.epochs + 1, 
                    time.time() - self.epochStartTime))
            for key in self.parNames:
                val = self.container[key][-1]
                f.write('  {}:\t\t{:.6f}\n'.format(key, val))
    def logStart(self):
        logPath = self.logPath
        with open(logPath, 'a') as f:
            f.write('Experiment {} started {}\n'.format(self.expId, currentTime()))
    def logEnd(self, success=True):
        result = 'SUCCEED' if success else 'FAILED'
        logPath = self.logPath
        with open(logPath, 'a') as f:
            f.write('Experiment {} {} at {}\n'.format(self.expId, result, currentTime()))
        self.success = success
        self.logGlobal()
    def logModel(self):
        #TODO
        pass
    def logGlobal(self):
        logPath = self.mainlogPath
        toSave = ['trainLoss', 'valLoss', 'valAcc']
        assert np.all([par in self.parNames for par in toSave])
        best = [np.min(self.container[key]) for key in toSave]
        bestStr = '{:.3f}, {:.3f}, {:.3f}'.format(best[0], best[1], best[2])
        with open(logPath, 'a') as f:
            f.write('{},{},{},{},{}\n'.format(self.expId, self.startDate, 
                    currentTime(), self.success, bestStr))
                
if __name__ == "__main__":
    logger = Logger('testNetwork')
    logger.logStart()
    for i in range(10):
        logger.processEpoch(np.random.rand(),np.random.rand(),np.random.rand())
    logger.logEnd()
#TODO - global log file for all experiments
#TODO - local log files
        
        