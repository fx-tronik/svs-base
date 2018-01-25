#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 07:13:22 2018
MPII dataset
@author: jakub
"""
import numpy as np
import os
import glob
import cv2
from multiprocessing import Process, Queue
from testDataset import testDataset
import time
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, dataDir, imageSize, targetSize, batchSize,
                 minLabelArea=256., maxUpscale=2., minUnderscale=0.25,
                 biOutput=True, networkScale = 8):
        self.dataDir = dataDir
        self.imageSize = imageSize
        self.targetSize = targetSize
        self.batchSize = batchSize
        self.minLabelArea = minLabelArea
        self.maxUpscale = maxUpscale
        self.minUnderscale = minUnderscale
        self.biOutput = biOutput
        self.networkScale = networkScale
        sCatNms = ['head']
        kpCatNms = ['r_ankle','r_knee','r_hip', 'l_hip', 'l_knee', 'l_ankle', 
                    'pelvis', 'thorax', 'upper_neck', 'head_top', 'r_wrist', 
                    'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
        self.sCatNms = sCatNms
        self.kpCatNms = kpCatNms
        self.initImages()
        self.initWorkers()
        
    def batchGenerator(self, queue, val=False):
        valIds = self.valIds
        trainIds = self.trainIds
        data = self.data
        dataDir = self.dataDir
        batchSize = self.batchSize
        imgsDir = os.path.join(dataDir, 'images')
        imgIds = valIds if val else trainIds
        imgNum = len(imgIds)
        imgIter = 0
        def popContainer(container, batchSize):
            data = container[:batchSize]
            container = container[batchSize:]
            return np.stack(data, axis = 0), container
        iCont, tCont, mCont = [], [], []
        while(1):
            #start = time.time()
            imgId = imgIds[imgIter]
            anns = data[imgId]
            imgPath = os.path.join(imgsDir, anns[0]['filename'])
            image = cv2.imread(imgPath, 0)
            image = image.astype(np.float32) / 255.0
            h, w  = image.shape
            iTargets = self.generateInstancesTargets(anns, h, w)
            kpTargets = self.generateKeypointsTargets(anns, h, w)
            targets = np.append(iTargets, kpTargets, axis = 0)
            masks = self.generateMask(targets)
            scale, stride = self.scaleStride(h, w)
            image = self.cutPatch(image, scale, stride)
            [targets, masks] = map(lambda t: self.cutPatch(t, scale, stride, alreadyNetworkScaled=True), 
                              [targets, masks])
            targets, masks = self.formatTarget(targets, masks, biOutput=self.biOutput)
            iCont.append(image)
            tCont.append(targets)
            mCont.append(masks)
            imgIter+=1
            if imgIter >= imgNum:
                imgIter = 0
            while (len(iCont) >= batchSize):
                iB, iCont = popContainer(iCont, batchSize)
                tB, tCont = popContainer(tCont, batchSize)
                mB, mCont = popContainer(mCont, batchSize)
                queue.put((iB, tB, mB))
                
    def cutPatch(self, I, scale=1.0, stride=[0, 0], alreadyNetworkScaled = False):
        networkScale = self.networkScale
        imageSize = self.imageSize
        tarSize = self.imageSize / networkScale
        if len(np.shape(I)) == 2:
            I = np.expand_dims(I, axis=0)
        patch = []
        sx = float(stride[1]) / scale
        sy = float(stride[0]) / scale
        beforeSize = float(imageSize) / scale
        if alreadyNetworkScaled:
            
            nsx = float(stride[1]) / (scale * networkScale)
            nsy = float(stride[0]) / (scale * networkScale)
            sBeforeSize = float(tarSize) / scale
            for ch in I:
                ch = ch[int(np.rint(nsx)):int(np.rint(nsx+sBeforeSize)), int(np.rint(nsy)):int(np.rint(nsy+sBeforeSize))]
                ch = cv2.resize(ch, (int(tarSize), int(tarSize)), interpolation=cv2.INTER_NEAREST)
                patch.append(ch)
        else:
            for ch in I:
                ch = ch[int(sx):int(sx+beforeSize), int(sy):int(sy+beforeSize)]
                ch = cv2.resize(ch, (int(imageSize), int(imageSize)), interpolation=cv2.INTER_NEAREST)
                patch.append(ch)
        return np.stack(patch)
                
    def endDataset(self):
        self.trainP.terminate()
        self.valP.terminate()
        self.trainQ = None
        self.valQ = None
        
    def formatTarget(self, target, mask, biOutput=True):
        #resize target and masks to final shape
        #biOutput - use if two neurons are coding output
        #inputs in format (classNum, imgSize, imgSize)
        targetSize = self.targetSize
        imageSize = self.imageSize
        networkScale = self.networkScale
        margin = (imageSize / networkScale - targetSize) / 2
        scaledMargin = (imageSize - targetSize * networkScale) /2
        target = target[:,margin:-margin, margin:-margin]
        mask = mask[:,margin:-margin, margin:-margin]
        if biOutput:
            target = np.stack([target, 1.0 - target], axis = 1)
        return target, mask
        
    def generateInstancesTargets(self, anns, h, w):
        networkScale = self.networkScale
        sCatNms = self.sCatNms
        ws, hs = [dim / networkScale for dim in [w, h]]
        targets = [np.zeros((hs, ws), dtype=np.float32) for _ in sCatNms]
        #targets = [np.zeros((h, w), dtype=np.float32) for _ in sCatNms]
        for ann in anns:
            sCatId = 0
            rectImg = np.zeros_like(targets[sCatId])
            x1, y1, x2, y2 = [int(np.rint(x / networkScale)) for x in ann['head_rect']]
            rectImg = cv2.rectangle(rectImg, (x1, y1), (x2, y2), 1.0, -1)
            targets[sCatId] += rectImg
        instTargets = np.stack(targets)
        return instTargets
    
    def generateKeypointsTargets(self, anns, h, w, tarDim=1):
        #generate target from person keypoints annotation
        #targets are point, but extended circles with radius tarDim
        kpCatNms = self.kpCatNms
        networkScale = self.networkScale
        ws, hs = [dim / networkScale for dim in [w, h]]
        targets = [np.zeros((hs, ws), dtype=np.float32) for _ in kpCatNms]
        for ann in anns:
            kps  = ann['joint_pos']
            kpsVis = ann['is_visible']
            for i, kpNm in enumerate(kpCatNms):
                if str(i) in kps.keys() and str(i) in kpsVis.keys():
                    kpP = np.rint(np.array(kps[str(i)]) / networkScale).astype(np.int)
                    kpVis = 0.5 * (kpsVis[str(i)] + 1.0) #target: 0.5 invisible, 1 visible
                    cv2.circle(targets[i], (kpP[0], kpP[1]), int(tarDim), kpVis, -1)
        kpTargets = np.stack(targets)
        return kpTargets        
    def generateMask(self, totalTargets):
        '''
        Simplified version (without borders)
        '''
        sCatNms = self.sCatNms
        kpCatNms = self.kpCatNms
        weights = np.ones(len(sCatNms) + len(kpCatNms))
        weights[:len(sCatNms)] = 20
        weights[-len(kpCatNms):] = 100
        masks = np.copy(totalTargets)
        masks *= weights.reshape((-1, 1, 1))
        return masks

    def initImages(self):
        #change with final dataset!
        dataDir = self.dataDir
        annFile = os.path.join(dataDir, 'data.json')
        data = []
        with open(annFile) as f:
            data = f.readlines()
        dicts = [eval(line) for line in data]
        images = glob.glob(os.path.join(dataDir, 'images', '*.jpg'))
        images = [os.path.basename(imPath) for imPath in images]
        data = []
        for img in images:
            instances = filter(lambda ann: ann['filename'] == img, dicts)
            if instances:
                data.append(instances)
            
        dataNum = len(data)
        indices = np.arange(dataNum)
        valDataNum = int(0.1 * dataNum)
        #all images with annotations are in train data (test data have no annotations)
        np.random.seed(7)
        np.random.shuffle(indices)
        valIds = indices[:valDataNum]
        trainIds = indices[valDataNum:]
        self.valIds = valIds
        self.trainIds = trainIds
        self.data = data
    def initWorkers(self):
        self.trainQ = None
        self.valQ = None
        self.trainQ = Queue(12)
        self.valQ = Queue(5)
        self.trainP = Process(target=self.batchGenerator, args=(self.trainQ, False))
        self.trainP.start()
        self.valP = Process(target=self.batchGenerator, args=(self.valQ, True))
        self.valP.start()
        
    def iterateMinibatches(self, val = False):
        valBatches = 1
        queue = self.valQ if val else self.trainQ
        batches = valBatches if val else 20 * valBatches
        for batchId in range(batches):
            yield queue.get()  
            
    def scaleStride(self, h, w):
        #generate random scale and stride to select patch from image
        #to select top left corner:
        scale = 1.0
        stride = (0, 0)
        minUnderscale = self.minUnderscale
        imageSize = self.imageSize
        scale = minUnderscale + (1.0 - minUnderscale)*np.random.rand()
          #Label area should be bigger than minLabelArea - 256 (16px x 16px)
        scale = np.max([scale] + [float(imageSize) / float(dim) for dim in [w, h]])
        wScaled, hScaled = [int(np.ceil(dim * scale)) for dim in [w, h]]
        stride = [np.random.randint(0, dim - imageSize+1) for dim in [wScaled, hScaled]]
        return scale, stride


        
        
if __name__ == "__main__":
    dataDir = '/home/jakub/data/MPII'
    dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                       batchSize=8, minLabelArea=16)
    #testDataset(dataset, classes = [6])
    
    start = time.time()
    for epoch in range(1):
        try:
            for inputs, targets, masks in dataset.iterateMinibatches(val=False):
                pass
    #==============================================================================
    #             for i in range(inputs.shape[0]):
    #                 dataset.showTensor(inputs[i])
    #==============================================================================
        except KeyboardInterrupt:
            dataset.endDataset()
    dataset.endDataset()
    print 'It took {} sconds'.format(time.time() - start)
