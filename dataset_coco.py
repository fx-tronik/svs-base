#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:01:20 2018
Plik przygotowujący obrazy ze zbioru COCO przy użyciu API i narzędzi Kerasa
@author: jakub
"""
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import cv2
import random
from multiprocessing import Process, Queue
import string
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
#cat - category
#img - image
#ann - annotation

class Dataset:
    def __init__(self, dataDir, imageSize, targetSize, batchSize,
                 minLabelArea=256., maxUpscale=2., minUnderscale=0.6):
        self.dataDir = dataDir
        self.imageSize = imageSize
        self.targetSize = targetSize
        self.batchSize = batchSize
        self.minLabelArea = minLabelArea
        self.maxUpscale = maxUpscale
        self.minUnderscale = minUnderscale
        self.initImages()
        self.initWorkers()

    def initImages(self):
        #change with final dataset!
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
        weights[sCatNms.index('person')] = 2.0
        weights[-len(kpCatNms):] = 2.0
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
        sCatNms = self.sCatNms
        w, h = img['width'], img['height']
        targets = [np.zeros((h, w), dtype=np.float32) for _ in sCatNms]
        for ann in anns:
            sCatId = ann['supercategory_id']
            targets[sCatId] += coco.annToMask(ann).astype(np.float32)
        instTargets = np.stack(targets)
        return instTargets

    def generateKeypointsTargets(self, img, anns, tarDim=2):
        #generate target from person keypoints annotation
        #targets are point, but extended circles with radius tarDim
        kpCatNms = self.kpCatNms
        w, h = img['width'], img['height']
        targets = [np.zeros((h, w), dtype=np.float32) for _ in kpCatNms]
        for ann in anns:
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                #sks = np.array(coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
                #kpn = coco_kps.loadCats(ann['category_id'])[0]['keypoints']
                kps = np.array(ann['keypoints'])
                x = kps[0::3]
                y = kps[1::3]
                v = kps[2::3]
                for i, (xCor, yCor, vis) in enumerate(zip(x, y, v)):
                    if np.all([xCor, yCor, vis]):
                        '''
                        vis = 0 -> not visible nor labeled
                        vis = 1 -> not visible but labeled
                        vis = 2 -> visible and labeled
                        '''
                        cv2.circle(targets[i], (xCor, yCor), tarDim, vis, -1)
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
        scale = minUnderscale + (maxUpscale - minUnderscale)*random.random()
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
        wScaled, hScaled = [int(dim * scale) for dim in [w, h]]
        stride = [np.random.randint(0, dim - imageSize+1) for dim in [wScaled, hScaled]]
        return scale, stride

    def cutPatch(self, I, scale=1.0, stride=[0, 0]):
        imageSize = self.imageSize
        if len(np.shape(I)) == 2:
            I = np.expand_dims(I, axis=0)
        patch = []
        for ch in I:
            ch = cv2.resize(ch, None, fx=scale, fy=scale)
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
        workingCocos = cocos[val::2] #instances and person keypoints
        dataType = types[val]
        imgIds = workingCocos[0].getImgIds()
        seed = 7
        random.seed(seed)
        random.shuffle(imgIds)
        imgIter = 0  
        def popContainer(container, batchSize):
            data = container[:batchSize]
            print 'BF', np.shape(container)
            container = container[batchSize:]
            return np.stack(data, axis = 0), container
        iCont, tCont, mCont = [], [], []
        while(1):
            imgId = imgIds[imgIter]
            img = workingCocos[0].loadImgs(imgId)[0] #only one image
            annIds = workingCocos[0].getAnnIds(imgIds=img['id'])
            anns = workingCocos[0].loadAnns(annIds)
            addSuperCat(anns)
                
            imgPath = os.path.join(dataDir, dataType, img['file_name'])
            image = cv2.imread(imgPath, 0)
            image = image.astype(np.float32) / 255.0
            
            iTargets = self.generateInstancesTargets(img, anns, workingCocos[0])
            kpAnns = []
            if personInAnns(anns):
                kpAnnIds = workingCocos[1].getAnnIds(imgIds=img['id'])
                kpAnns = workingCocos[1].loadAnns(kpAnnIds)
            kpTargets = self.generateKeypointsTargets(img, kpAnns)
            targets = np.append(iTargets, kpTargets, axis = 0)
            masks = self.generateMask(targets)
            
            scale, stride = self.scaleStride(img, anns)
            [image, targets, masks] = map(lambda t: self.cutPatch(t, scale, stride), 
                                          [image, targets, masks])
            iCont.append(image)
            tCont.append(targets)
            mCont.append(masks)
            imgIter+=1
            while (len(iCont) >= batchSize):
                iB, iCont = popContainer(iCont, batchSize)
                tB, tCont = popContainer(tCont, batchSize)
                mB, mCont = popContainer(mCont, batchSize)
                queue.put((iB, tB, mB))

    def initWorkers(self):
        self.trainQ = None
        self.valQ = None
        self.trainQ = Queue(12)
        self.valQ = Queue(5)
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
        valBatches = 1
        queue = self.valQ if val else self.trainQ
        batches = valBatches if val else 20 * valBatches
        for batchId in range(batches):
            yield queue.get()
        
if __name__ == "__main__":
    dataDir = '/home/jakub/data/fxi/coco'
    dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=40, 
                       batchSize=8)
    try:
        for inputs, targets, masks in dataset.iterateMinibatches(val=False):
            print inputs.item(0)
            for i in range(inputs.shape[0]):
                dataset.showTensor(inputs[i])
    except KeyboardInterrupt:
        dataset.endDataset
    dataset.endDataset()

#TODO BUG - > repeating images
