#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:01:20 2018
Plik przygotowujący obrazy ze zbioru COCO przy użyciu API 
@author: jakub
"""
import matplotlib
matplotlib.use('AGG')
from pycocotools.coco import COCO
#from pycocotools import mask as maskUtils
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import cv2
import random
from multiprocessing import Process, Queue
import time
import string
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
#cat - category
#img - image
#ann - annotation

class Dataset:
    def __init__(self, dataDir, imageSize, targetSize, batchSize,
                 minLabelArea=256., maxUpscale=2., minUnderscale=0.6,
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
        self.initImages()
        self.initWorkers()

    def initImages(self):
        #change with final dataset!
        dataDir = self.dataDir
        trainType, valType = 'val2017', 'val2017'
        types = [trainType, valType]
        annFiles = ['{}/annotations/instances_{}.json'.format(dataDir, dataType) \
                    for dataType in types]
        annFiles += ['{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType) \
                    for dataType in types]
        cocos = [COCO(annFile) for annFile in annFiles]
        cats = cocos[0].loadCats(cocos[0].getCatIds())
        sCatNms = sorted(list(set([cat['supercategory'] for cat in cats])))
        kpCatNms = cocos[2].loadCats(1)[0]['keypoints']
        weights = np.ones(29) #class weights (person and keypoints more important)
        weights[sCatNms.index('person')] = 10
        weights[-len(kpCatNms):] = 50
        borders = [False for _ in range(29)]
        borders[sCatNms.index('person')] = 'strict'
        borders[-len(kpCatNms):] = ['fuzzy' for _ in range(len(kpCatNms))]
        self.sCatNms = sCatNms
        self.kpCatNms = kpCatNms
        self.cocos = cocos
        self.types = types
        self.cats = cats
        self.weights = weights
        self.borders = borders
    def addSuperCat(self, anns):
        cats = self.cats
        for ann in anns:
            if not 'category_id' in ann:
                raise ValueError("Wrong format of annotation")
            annSupCat = [cat['supercategory'] for cat in cats \
                         if cat['id'] == ann['category_id']][0]
            ann['supercategory_id'] = self.sCatNms.index(annSupCat)


    def generateInstancesTargets(self, img, anns, coco):
        networkScale = self.networkScale
        sCatNms = self.sCatNms
        kpCatNms = self.kpCatNms
        w, h = img['width'], img['height']
        ws, hs = [dim / networkScale for dim in [w, h]]
        targets = [np.zeros((hs, ws), dtype=np.float32) for _ in sCatNms]
        kpMasks = np.zeros((hs, ws), dtype=np.float32) 
        #targets = [np.zeros((h, w), dtype=np.float32) for _ in sCatNms]
        for ann in anns:
            sCatId = ann['supercategory_id']
            try:
                ann['segmentation'] = map(lambda x: [el / networkScale for el in x], ann['segmentation'])
                tempMask = coco.annToMask(ann).astype(np.float32)[:hs,:ws]
                if ann['iscrowd'] == 1:
                    tempKpMask = np.copy(tempMask)
            except TypeError:
               tempMask = coco.annToMask(ann).astype(np.float32)
               tempMask = cv2.resize(tempMask,(ws, hs), interpolation=cv2.INTER_NEAREST)
               tempKpMask = np.copy(tempMask)
            targets[sCatId] += tempMask#coco.annToMask(ann).astype(np.float32)[:hs,:ws]
            if ann['iscrowd'] == 1:
                kpMasks += tempKpMask
        instTargets = np.stack(targets)
        kpMasks = 1.0 - np.stack(len(kpCatNms) * [kpMasks])
        return instTargets, kpMasks

    def generateKeypointsTargets(self, img, anns, tarDim=1):
        #generate target from person keypoints annotation
        #targets are point, but extended circles with radius tarDim
        kpCatNms = self.kpCatNms
        networkScale = self.networkScale
        w, h = img['width'], img['height']
        ws, hs = [dim / networkScale for dim in [w, h]]
        targets = [np.zeros((hs, ws), dtype=np.float32) for _ in kpCatNms]
        for ann in anns:
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                #sks = np.array(coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
                #kpn = coco_kps.loadCats(ann['category_id'])[0]['keypoints']
                tarDim = np.sqrt(ann['area']) / 10 / networkScale 
                kps = np.array(ann['keypoints'])
                x = kps[0::3] / networkScale
                y = kps[1::3] / networkScale
                v = kps[2::3]
                for i, (xCor, yCor, vis) in enumerate(zip(x, y, v)):
                    if np.all([xCor, yCor, vis]):
                        '''
                        vis = 0 -> not visible nor labeled
                        vis = 1 -> not visible but labeled
                        vis = 2 -> visible and labeled
                        '''
                        cv2.circle(targets[i], (xCor, yCor), int(tarDim), vis, -1)
        kpTargets = np.stack(targets)
        return kpTargets          

    def generateMask(self, totalTargets, borderW=5, 
                     strictBMlp=4.0, fuzzyBMlp=0.0):
        '''
        strictBorder - weight on border more important than inside of instance
                       borders more precise, harder to train
        fuzzyBorder - zero weight on border - easier to train
        '''
        weights = self.weights
        borders = self.borders
        masks = np.copy(totalTargets)
        masks *= weights.reshape((-1, 1, 1))
        kernel = np.ones((borderW, borderW), np.float32)
        for mask, border in zip(masks, borders):
            if border:
                dilation = cv2.dilate(mask, kernel, iterations=1)
                outBorder = dilation - mask
            mask[mask == 0] = 1.0
            if border == 'strict':
                mask[outBorder > 0] = strictBMlp
            elif border == 'fuzzy':
                mask[outBorder > 0] = fuzzyBMlp
        return masks
        
    def scaleStride(self, img, anns):
        #generate random scale and stride to select patch from image
        #to select top left corner:
        scale = 1.0
        stride = (0, 0)
        maxUpscale = self.maxUpscale
        minUnderscale = self.minUnderscale
        minLabelArea = self.minLabelArea
        imageSize = self.imageSize
        scale = minUnderscale + (scale - minUnderscale)*random.random()
        w, h = img['width'], img['height']
        labelAreas = [ann['area'] for ann in anns]
        if labelAreas:
            minImgLabel = np.min(labelAreas)
        else:
            minImgLabel = 10 * minLabelArea
        #Label area should be bigger than minLabelArea - 256 (16px x 16px)
        scale = max([scale, minLabelArea / minImgLabel])
        scale = np.min([scale, maxUpscale])
        scale = np.max([scale] + [float(imageSize) / float(dim) for dim in [w, h]])
        wScaled, hScaled = [int(np.ceil(dim * scale)) for dim in [w, h]]
        stride = [np.random.randint(0, dim - imageSize+1) for dim in [wScaled, hScaled]]
        return scale, stride
    
    def cutPatch2(self, I, scale=1.0, stride=[0, 0], alreadyNetworkScaled = False):
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
            

    def cutPatch(self, I, scale=1.0, stride=[0, 0]):
        imageSize = self.imageSize
        if len(np.shape(I)) == 2:
            I = np.expand_dims(I, axis=0)
        patch = []
        for ch in I:
            ch = cv2.resize(ch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ch = ch[stride[1]:stride[1]+imageSize, stride[0]:stride[0]+imageSize]
            patch.append(ch)
        patch = np.stack(patch)
        return patch
    def batchGenerator(self, queue, val=False):
        cocos = self.cocos
        dataDir = self.dataDir
        types = self.types
        addSuperCat = self.addSuperCat
        personInAnns = self.personInAnns
        batchSize = self.batchSize
        kpCatNms = self.kpCatNms
        workingCocos = cocos[val::2] #instances and person keypoints
        dataType = types[val]
        imgIds = workingCocos[0].getImgIds()
        seed = 7
        random.seed(seed)
        random.shuffle(imgIds)
        imgIter = 0  
        def popContainer(container, batchSize):
            data = container[:batchSize]
            container = container[batchSize:]
            return np.stack(data, axis = 0), container
        iCont, tCont, mCont = [], [], []
        while(1):
            #start = time.time()
            imgId = imgIds[imgIter]
            img = workingCocos[0].loadImgs(imgId)[0] #only one image
            annIds = workingCocos[0].getAnnIds(imgIds=img['id'])
            anns = workingCocos[0].loadAnns(annIds)
            addSuperCat(anns)
            #print 'COCOS processing took {} s'.format(time.time() - start)
            #start = time.time()
            imgPath = os.path.join(dataDir, dataType, img['file_name'])
            image = cv2.imread(imgPath, 0)
            image = image.astype(np.float32) / 255.0
            #print 'LOAD image took {} s'.format(time.time() - start)
            
            iTargets, kpMasks = self.generateInstancesTargets(img, anns, workingCocos[0])
            kpAnns = []
            if personInAnns(anns):
                kpAnnIds = workingCocos[1].getAnnIds(imgIds=img['id'])
                kpAnns = workingCocos[1].loadAnns(kpAnnIds)
            kpTargets = self.generateKeypointsTargets(img, kpAnns)
            targets = np.append(iTargets, kpTargets, axis = 0)
            masks = self.generateMask(targets)
            masks[-len(kpCatNms):] *= kpMasks 
            scale, stride = self.scaleStride(img, anns)
            start = time.time()
            image = self.cutPatch2(image, scale, stride)
            [targets, masks] = map(lambda t: self.cutPatch2(t, scale, stride, alreadyNetworkScaled=True), 
                                          [targets, masks])
            #print 'Targets generating took {} s'.format(time.time() - start)
            targets, masks = self.formatTarget(targets, masks, biOutput=self.biOutput)

            iCont.append(image)
            tCont.append(targets)
            mCont.append(masks)
            imgIter+=1
            if imgIter >= len(imgIds):
                imgIter = 0
            while (len(iCont) >= batchSize):
                iB, iCont = popContainer(iCont, batchSize)
                tB, tCont = popContainer(tCont, batchSize)
                mB, mCont = popContainer(mCont, batchSize)
                queue.put((iB, tB, mB))

    def initWorkers(self):
        self.trainQ = None
        self.valQ = None
        self.trainQ = Queue(8)
        self.valQ = Queue(3)
        self.trainP = Process(target=self.batchGenerator, args=(self.trainQ, False))
        self.trainP.start()
        self.valP = Process(target=self.batchGenerator, args=(self.valQ, True))
        self.valP.start()
    def endDataset(self):
        self.trainP.terminate()
        self.valP.terminate()
        self.trainQ = None
        self.valQ = None
    def personInAnns(self, anns):
        #check if there is person in annotations -> then use keypoints anns
        for ann in anns:
            if ann['category_id'] == 1:
                return True
        return False #if no person in annotations
    def showTensor(self, t):
        if len(t.shape) == 3:
            tShow = np.sum(t, axis=0)
        elif len(t.shape) == 2:
            tShow = np.copy(t)
        else:
            print 'Invalid tensor format'
            return False
        chrs = list(string.ascii_lowercase)
        random.shuffle(chrs)
        wId = ''.join(chrs[:5])
        winName = 'test_{}'.format(wId)
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.imshow(winName, tShow)
        cv2.waitKey(0)
        cv2.destroyWindow(winName)
    def iterateMinibatches(self, val = False):
        valBatches = 8
        queue = self.valQ if val else self.trainQ
        batches = valBatches if val else 8 * valBatches
        for batchId in range(batches):
            yield queue.get()
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
#==============================================================================
#         target, mask = [np.stack([cv2.resize(ch, None, fx=1.0 / networkScale, \
#                                              fy=1.0 / networkScale, \
#                                              interpolation=cv2.INTER_NEAREST)[margin:margin+targetSize, margin:margin+targetSize]\
#                                 for ch in tensor]) for tensor in [target, mask]]
#==============================================================================
        #another version, same time       
#==============================================================================
#         target, mask = [np.stack([cv2.resize(ch[scaledMargin:-scaledMargin, scaledMargin:-scaledMargin],\
#                              (targetSize, targetSize),   
#                              interpolation=cv2.INTER_NEAREST)\
#                 for ch in tensor]) for tensor in [target, mask]]
#==============================================================================
        target[target > 0.0] = 1.0
        if biOutput:
            target = np.stack([target, 1.0 - target], axis = 1)
        return target, mask
    
def testDataset(dataset, classes=[0]):
    catNms = dataset.sCatNms + dataset.kpCatNms
    def flattenImage(inputs, targets, masks, cl, bId):
        scale=8
        inImage = inputs[bId,0]
        imSize = inImage.shape[0]
        maImage = masks[bId, cl]
        taImage = targets[bId, cl, 0]
        taSize = taImage.shape[0]
        margin = (imSize - (taSize * scale)) / 2
        #if np.max(taImage) > 1.0:
        #    taImage /= np.max(taImage)
        #if np.max(maImage) > 1.0: 
        #    maImage /= np.max(maImage)
        maImage = cv2.resize(maImage, None, fx=scale, fy=scale)
        taImage = cv2.resize(taImage, None, fx=scale, fy=scale)
        maImageCanvas = np.zeros((imSize, imSize), dtype=np.float32)
        taImageCanvas = np.zeros((imSize, imSize), dtype=np.float32)
        taImageCanvas[margin:-margin, margin:-margin] = taImage
        maImageCanvas[margin:-margin, margin:-margin] = maImage
        image1 = 0.5 * inImage + 0.5 * taImageCanvas
        image2 = 0.5 * inImage + 0.5 * maImageCanvas
        return image1, image2
    def showImage(im1, im2, cl, bId):
        winTarName = 'targets_{}'.format(catNms[cl])
        winMasName = 'masks_{}'.format(catNms[cl])
        cv2.namedWindow(winTarName, cv2.WINDOW_NORMAL)
        cv2.namedWindow(winMasName, cv2.WINDOW_NORMAL)
        cv2.imshow(winTarName, im1)
        cv2.imshow(winMasName, im2)
        cv2.moveWindow(winMasName, 1000, 1000)
        cv2.waitKey(0)
        cv2.destroyWindow(winTarName)
        cv2.destroyWindow(winMasName)

    for cl in classes:
        for inputs, targets, masks in dataset.iterateMinibatches(val=False):
            for bId in range(inputs.shape[0]):
                if np.max(targets[bId, cl,0]) > 0.0:
                    im1, im2 = flattenImage(inputs, targets, masks, cl, bId)
                    showImage(im1, im2, cl, bId)
                
if __name__ == "__main__":
    dataDir = '/home/jakub/data/coco'
    dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                       batchSize=8, minLabelArea=16)
    #testDataset(dataset, classes = [17])
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

