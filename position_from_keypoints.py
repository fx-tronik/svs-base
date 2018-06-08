#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:10:55 2018

@author: jakub
"""
from __future__ import print_function
import numpy as np
#from common import CocoPart
from enum import Enum

test_keypoints = [503, 231, 2, 514, 214, 2, 500, 216, 2, 547, 208, 2, 0, 0, 0, 
                  617, 251, 2, 533, 287, 2, 679, 333, 2, 558, 383, 2, 
                  689, 414, 2, 568, 434, 2, 659, 454, 2, 588, 464, 2, 
                  682, 586, 2, 595, 596, 2, 0, 0, 0, 0, 0, 0]

PartPairs = {
        'LArm': (5, 7),
        'RArm': (6, 8),
        'LFarm': (7, 9),
        'RFarm': (8, 10),
        'ShLine': (5, 6),
        'HLine': (11, 12),
        'LThigh': (11, 13),
        'RThigh': (12, 14),
        'LCalf': (13, 15),
        'RCalf': (14, 16), 
        'LBody': (5, 11),
        'RBody': (6, 12)
        }

class BaseDirections(Enum):
    Front = 0
    Back = 1
    Left = 2
    Right = 3
    Unknown = 4

def pair_exist(keypoints, pair_name='LArm'):
    pair_s, pair_f = PartPairs[pair_name]
    return keypoints[3 * pair_s + 2] != 0  and keypoints[3 * pair_f + 2] != 0

def pair_angle(keypoints, pair_name = 'LArm'):
    #direction in radians, 0 is downward, counter clockwise (CCW)
    if not pair_exist(keypoints, pair_name):
        return None
    pair_s, pair_f = PartPairs[pair_name]
    xs, ys = keypoints[3 * pair_s], keypoints[3 * pair_s + 1]
    xf, yf = keypoints[3 * pair_f], keypoints[3 * pair_f + 1]
    dx = xf - xs
    dy = yf - ys
    angle = np.arctan2(dx, dy)
    if angle < 0.0:
        angle += 2 * np.pi
    return angle

def is_horizontal(angle):
    if not angle: #if pair not found
        return False
    margin = 10 #in deg
    deg_angle = np.rad2deg(angle)
    return (90 - margin < deg_angle < 90 + margin) or (270 - margin < deg_angle < 270 + margin)

def is_vertical(angle):
    if not angle:
        return False
    margin = 10 #in deg
    deg_angle = np.rad2deg(angle)
    return ( - margin < deg_angle <  margin) or (180 - margin < deg_angle < 180 + margin)

def is_positive(angle):
    #return true if is directed right or down
    if not angle:
        return False
    margin = 10
    deg_angle = np.rad2deg(angle)
    return  (- margin < deg_angle <  margin) or (90 - margin < deg_angle < 90 + margin)

def is_negative(angle):
    #return true if is directed left or up
    if not angle:
        return False
    margin = 10
    deg_angle = np.rad2deg(angle)
    return  (180 - margin < deg_angle <  180 + margin) or (270 - margin < deg_angle < 270 + margin)

def get_base_direction(keypoints):
    #human is visible from Front, Back or side (L? R?)
    direction = BaseDirections.Unknown
    if is_horizontal(pair_angle(keypoints, 'HLine')) or \
    is_horizontal(pair_angle(keypoints, 'ShLine')):
        if is_positive(pair_angle(keypoints, 'HLine')) or \
        is_positive(pair_angle(keypoints, 'ShLine')):
            direction = BaseDirections.Back
        elif is_negative(pair_angle(keypoints, 'HLine')) or \
        is_negative(pair_angle(keypoints, 'ShLine')):
            direction = BaseDirections.Front
    elif pair_exist(keypoints, 'LArm') and not pair_exist (keypoints, 'RArm'):
        direction = BaseDirections.Left
    elif pair_exist(keypoints, 'RArm') and not pair_exist (keypoints, 'LArm'):
        direction = BaseDirections.Right
    
    return direction

def get_network_input(keypoints):
    '''
    vector of length 28
    0:4 front back left right one code
    4:28 sin cos of each point pair in PartPairs
    '''
    direction = get_base_direction(keypoints)
    dir_locus = np.zeros(4)
    dir_locus[direction.value] = 1.0
    trigs = np.zeros(24)
    for i, pair in enumerate(sorted(PartPairs.keys())):
        angle = pair_angle(keypoints, pair)
        if angle:
            trigs[2 * i] = np.sin(angle)
            trigs[2 * i +1] = np.cos(angle)
        else:
            trigs[2 * i] = 0.0
            trigs[2 * i +1] = 1.0
    net_input = np.concatenate([dir_locus, trigs])
    return net_input
            


if __name__ == '__main__':      
    direction = get_base_direction(test_keypoints)
    print(direction.name)
    