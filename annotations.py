#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:47:06 2018
Adnotacje do etykieciarki
@author: jakub
"""
from __future__ import print_function
import numpy as np
import json
import os
from common import Human, BodyPart, prepare_image_record
from position_from_keypoints import get_network_input

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
def search_humans(bbox_id, list_of_humans):
    return [i for i, element in enumerate(list_of_humans) if element.bbox_id == bbox_id]
class annotations:
    def __init__(self):
        data = []
        humans = []
        images = []
        curAnnId = MIN_ID
        self.data = data
        self.curAnnId = curAnnId
        self.humans = humans
        self.images = images
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
               'image_id': img_id, 'num_keypoints': 0, 'iscrowd': 0}
        if bboxId not in [anx['id'] for anx in self.data]:
            self.data.append(ann)
            print('Dodano nowy ann', tlx, tly, brx, bry)
        else:
            idToOverwrite = search_dictionaries('id', bboxId, self.data)            
            assert len(idToOverwrite) == 1
            self.data[idToOverwrite[0]] = ann
            print('Nadpisano ann w boksie {}'.format(bboxId))
        self.curAnnId = bboxId
        self.humans.append(Human(bboxId))
    def delBBox(self, bboxId):
        del_index = search_dictionaries('id', bboxId, self.data)
        if del_index:
            del(self.data[del_index[0]])
        else:
            raise ValueError('bboxID: {} does not exists'.format(bboxId))
        del_human_index = search_humans(bboxId, self.humans)
        if del_human_index:
            del(self.humans[del_human_index[0]])
        else:
            raise ValueError('Human with bboxID: {} does not exists'.format(bboxId))
        
        
    def newImage(self, path, image, imgId):
        rec = prepare_image_record(path, image, imgId)
        self.images.append(rec)
        
    def addBodyPart(self, bodyPart, bboxId):
        #find human with matching bboxId
        hid = search_humans(bboxId, self.humans)
        annid = search_dictionaries('id', bboxId, self.data)
        if len(hid) == 0 or len(annid) == 0:
            raise ValueError('FATAL ERROR, no human with matching bbox_id {} or {} found'.format(hid, annid))
        else:
            hid, annid = hid[0], annid[0]
        self.humans[hid].add_body_part(bodyPart)
        self.data[annid]['keypoints'][3*bodyPart.part_idx] = bodyPart.x
        self.data[annid]['keypoints'][3*bodyPart.part_idx + 1] = bodyPart.y
        self.data[annid]['keypoints'][3*bodyPart.part_idx + 2] = 2
        self.data[annid]['num_keypoints'] +=1
            
    def removeBodyPart(self, bodyPartId, bboxId):
        #find human with matching bboxId
        hid = search_humans(bboxId, self.humans)
        annid = search_dictionaries('id', bboxId, self.data)
        if len(hid) == 0 or len(annid) == 0:
            raise ValueError('FATAL ERROR, no human with matching bbox_id {} or {} found'.format(hid, annid))
        else:
            hid, annid = hid[0], annid[0]
        self.humans[hid].remove_body_part(bodyPartId)
        self.data[annid]['keypoints'][3*bodyPartId] = 0
        self.data[annid]['keypoints'][3*bodyPartId + 1] = 0
        self.data[annid]['keypoints'][3*bodyPartId + 2] = 0
        self.data[annid]['num_keypoints'] +=1
        
    def addAction(self, actionId, bboxId):
        annid = search_dictionaries('id', bboxId, self.data)
        if len(annid) == 0:
            raise ValueError('FATAL ERROR, no human with matching bbox_id {} or {} found'.format(annid))
        else:
            annid = annid[0]
        self.data[annid]['action'] = actionId
        
        
    def getBboxIds(self, img_id):
        ann_ids = []
        for ann in self.data:
            if int(ann['image_id']) == int(img_id):
                ann_ids.append(ann['id'])
        return ann_ids
    
    def imgInAnns(self, img_id):
        inSet = False
        for img in self.images:
            if img['id'] == img_id:
                inSet = True
                break
        return inSet
    
    def dumpJson(self, jsonFile):
        dump = {'annotations': self.data,
                'images': self.images}
        with open(jsonFile, 'w') as f:
            json.dump(dump, f)
            
    def loadJson(self, jsonFile):
        with open(jsonFile, 'r') as f:
            jsonData = json.load(f)
        self.data = jsonData['annotations']
        self.images = jsonData['images']
        self.humans = self.anns2humans(self.data)
        annId, imgId  = self.getMaxIds()
        self.setCurAnnId(annId)
        return annId, imgId
        
    def getMaxIds(self):
        #return max annotations id and img id
        maxAnnId, maxImgId = 0, 0
        for ann in self.data:
            maxAnnId = np.max([maxAnnId, int(ann['id'])])
            maxImgId = np.max([maxImgId, int(ann['image_id'])])
        return maxAnnId, maxImgId
    
    def get_action_data(self):
        xs = []
        ys = []
        for ann in self.data:
            action_id = ann.get('action', None)
            if action_id is not None:
                xs.append(get_network_input(ann.get('keypoints')))
                ys.append(action_id)
        return np.stack(xs, axis = 0), np.array(ys)
                
            
    
    @staticmethod
    def anns2humans(anns):
        humans = []
        for ann in anns:
            human = Human(ann['id'])
            for i in range(17):
                if ann['keypoints'][3*i + 2] !=0 : 
                    bP = BodyPart(ann['id'], i, ann['keypoints'][3*i], ann['keypoints'][3*i + 1], 2)
                    human.add_body_part(bP)
            humans.append(human)
        return humans
    

            
        
        
if __name__ == '__main__':
#==============================================================================
#     anns = annotations()
#     print anns.getAnnsNum()     
#     anns.newBBox(100,200,200,100)
#     print 'Aktualny BBOX', anns.getCurAnnId()
#     anns.newBBox(100,200,200,100, 0)
#     print 'Aktualny BBOX', anns.getCurAnnId()
#     anns.newBBox(100,200,200,100, 7)
#     print 'Aktualny BBOX', anns.getCurAnnId()
#     data = anns.data
#==============================================================================
    
    data = read_coco_anns(datasetType='val')
    annotations_ids = [x['id'] for x in data['annotations'] ] #min  183014 #max 900100581904
    images_id= [x['image_id'] for x in data['annotations'] ]  #min 36 #max 581921
