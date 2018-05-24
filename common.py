#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:10:18 2018

@author: jakub
"""
from enum import Enum
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import datetime
from position_from_keypoints import get_base_direction

class PosePart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18
    
class CocoPart(Enum):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16


    
CocoPairs = [
    (0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (5, 11), 
    (6, 12), (7, 9), (8, 10), (11, 12), (11,13), (12, 14), (13, 15), (14, 16)
]   

PosePairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)
    
    
class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list')

    def __init__(self, bbox_id):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        self.bbox_id = bbox_id

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])
    def add_body_part(self, bodyPart):
        self.body_parts[bodyPart.part_idx] = bodyPart
    def remove_body_part(self, bodyPartId):
        if bodyPartId in self.body_parts.keys():
            del self.body_parts[bodyPartId]

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])
    
def draw_humans(npimg, humans, bbox_ids, margin, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            if  human.bbox_id not in bbox_ids:
                continue
            # draw point
            for i in range(len(CocoPart)):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x + 0.5)+margin, int(body_part.y + 0.5)+margin)
                centers[i] = center
                npimg = cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

        return npimg
    
def draw_anns(img, anns, img_id, margin, bbox_id):
    draw_pos = True
    color = (0, 255, 0)
    for ann in anns:
        if int(ann['image_id']) != int(img_id):
            continue
        else:
            bbox = ann['bbox']
            color = (0, 0, 255) if ann['id'] == bbox_id else (0, 255, 0)
            bbox = [int(np.round(x)) for x in bbox]
            img = cv2.rectangle(img, (bbox[0] + margin, bbox[1] + margin),
                                (bbox[0] + bbox[2] + margin, bbox[1]+bbox[3]+margin), 
                                color, 2)
            if draw_pos:
                direction = get_base_direction(ann['keypoints'])
                img = cv2.putText(img, direction.name, 
                                  (bbox[0] + margin,  bbox[1]+ margin),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                  2, lineType=cv2.LINE_AA)
            
    return img

def prepare_dir(target_dir):
    images_dir = os.path.join(target_dir, 'images')
    anns_dir = os.path.join(target_dir, 'annotations')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(anns_dir):
        os.makedirs(anns_dir)

def prepare_image_record(path, image, img_id):
    u = unicode
    h, w, _ = image.shape
    rec = {'coco_url': u(os.path.basename(path)),
           'date_captured': u(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
           'file_name': u('{}.jpg'.format(img_id)), 
           'flickr_url': u(''),
           'height': h,
           'id': img_id,
           'license': 7,
           'width': w
           }
    return rec
