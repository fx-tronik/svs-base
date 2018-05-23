#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:58:30 2018

@author: jakub
"""
from __future__ import print_function
import os
import numpy as np
from annotations import annotations, MIN_ID, MIN_IMG_ID


def get_img_ids(anns):
    return [int(x['id']) for x in anns.images]
def get_bbox_ids(anns):
    return [int(x['id']) for x in anns.data]
def merge(anns1, anns2):
    img_ids1, img_ids2 = map(get_img_ids, [anns1, anns2])
    common_ids = list(set(img_ids1).intersection(img_ids2))
    if common_ids:
        print('WARNING: Same image ids for two datasets', common_ids)
        
    bbox_ids1, bbox_ids2 = map(get_bbox_ids, [anns1, anns2])
    new_id = np.max(bbox_ids1) + 1
    anns1.images += anns2.images
    for bbox in anns2.data:
        bbox['id'] = new_id
        new_id += 1
        anns1.data.append(bbox)
    return anns1
        
if __name__ == "__main__":
    TARGET_DIR = os.path.expanduser('~/data/OWN_COCO/')
    TARGET_DIR_2 = os.path.expanduser('~/data/NEW_COCO/')
    
    jsonFile = 'own_pose_dataset.json'
    
    anns1 = annotations()
    anns2 = annotations()
    
    ann_id1 , img_id1 = anns1.loadJson(os.path.join(TARGET_DIR_2, 'annotations', 'person_keypoints_train2017.json'))
    ann_id2 , img_id2 = anns2.loadJson(os.path.join(TARGET_DIR, 'annotations', jsonFile))
    
    anns = merge(anns1, anns2)
    anns.dumpJson(os.path.join(TARGET_DIR, 'annotations', 'pose_dataset.json'))

