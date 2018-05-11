#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:47:06 2018
Adnotacje do etykieciarki
@author: jakub
"""

import numpy as np
import json
import os

MIN_ID = 1000000000000 #to not colide with COCO
MIN_IMG_ID = 200000
n = 0
def read_coco_anns(datasetType = 'val', year = '2017'):
    rootDir = os.path.expanduser('~/data/coco/annotations')
    annFile = os.path.join(rootDir, 'person_keypoints_{}{}.json'.format(datasetType, year))
    with open(annFile, 'r') as f:
        dataDir = json.load(f)
    return dataDir
def search_dictionaries(key, value, list_of_dictionaries):
    return [i for i, element in enumerate(list_of_dictionaries) if element[key] == value]
class annotations:
    def __init__(self):
        data = []
        curAnnId = MIN_ID
        self.data = data
        self.curAnnId = curAnnId
    def getCurAnnId(self):
        return self.curAnnId
    def setCurAnnId(self, curAnnId):
        self.curAnnId = curAnnId
    def getAnnsNum(self):
        return len(self.data)
    def newBBox(self, sx, sy, x, y, bboxId = None, img_id = None):
        tlx = np.min([x, sx])
        tly = np.min([y, sy])
        brx = np.max([x, sx])
        bry = np.max([y, sy])
        if bboxId is None:
            bboxId = MIN_ID + len(self.data)
        ann = {'bbox':[tlx, tly, brx - tlx, bry-tly], 'id': bboxId, 'keypoints':51 * [0],
               'image_id': img_id}
        if bboxId not in [anx['id'] for anx in self.data]:
            self.data.append(ann)
            print 'Dodano nowy ann', tlx, tly, brx, bry
        else:
            idToOverwrite = search_dictionaries('id', bboxId, self.data)            
            assert len(idToOverwrite) == 1
            self.data[idToOverwrite[0]] = ann
            print 'Nadpisano ann w boksie {}'.format(bboxId)
        self.curAnnId = bboxId
        

if __name__ == '__main__':
    anns = annotations()
    print anns.getAnnsNum()     
    anns.newBBox(100,200,200,100)
    print 'Aktualny BBOX', anns.getCurAnnId()
    anns.newBBox(100,200,200,100, 0)
    print 'Aktualny BBOX', anns.getCurAnnId()
    anns.newBBox(100,200,200,100, 7)
    print 'Aktualny BBOX', anns.getCurAnnId()
    data = anns.data
    
    data = read_coco_anns(datasetType='val')
    annotations_ids = [x['id'] for x in data['annotations'] ] #min  183014 #max 900100581904
    images_id= [x['image_id'] for x in data['annotations'] ]  #min 36 #max 581921
