# -*- coding: utf-8 -*-
"""
Spyder Editor

cctv_disparity( ) - compute disparity from two tif files
13.02.19 - now uses same packages as cctv_processVideo( )
@filename: cctv_disparity.py
@author: mark.fisher@uea.ac.uk
@last_updated: 13.02.19
"""
import numpy as np
import cv2
import argparse
import os
from stereo import stereo_utils
from belt import belt_travel as bt
from utils import image_plotting as ip

global minDisp
minDisp = -1
global numDisp
numDisp = 16

def proportion_of_matched_pixels(disp, minDisp, numDisp):
    h, w = disp.shape
    D = disp[:,numDisp+minDisp:max([minDisp+w, w])]
    loc = np.where(D > (minDisp-1)*16)
    
    n_matched_pix = np.size(loc[0])
    n_total_pix = D.size
    
    return n_matched_pix / n_total_pix

def parse_args():
    
    parser = argparse.ArgumentParser(description='compute stereo disparity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, default="../data/",
                        help='Root pathname.')
    parser.add_argument('--nameL', type=str, default="beltE57.tif",
                        help='Left image filename.')
    parser.add_argument('--nameR', type=str, default="beltE55.tif",
                        help='Right image filename.')
    args = parser.parse_args()
    
    return args
    
    
if __name__ == '__main__':
    """
    Test stub for cctvDisparity
    """    
    args = parse_args()
    
    # creates a temporary directory to save data generated at runtime
    temp_path = args.root_path+'temp/'
    try:
        os.makedirs(temp_path)
    except OSError:
        if os.path.isdir(temp_path):
            pass
        
    # read images
    rawL = cv2.imread(args.root_path+args.nameL)
    rawR = cv2.imread(args.root_path+args.nameR)  
    
    dx, __ = bt.getBeltMotionByTemplateMatching(rawR, rawL) 
        
    imgL, imgR = stereo_utils.stereoPreprocess(rawL, rawR)
        
    cv2.imshow('rawL', rawL)
    cv2.imshow('rawR', rawR)
    cv2.imshow('imgL', imgL)
    cv2.imshow('imgR', imgR)
    
    #  compute disparity          
    dispL, __, __, __ = stereo_utils.findDisparity(ip.translateImg(imgL, (-dx, 0)), imgR, minDisp=minDisp, numDisp=numDisp)
 
  
    p = proportion_of_matched_pixels(dispL, minDisp, numDisp)
    print('Fraction of Matched Pixels = {0:0.2f}'.format(p))
  
    # display dispaity
    vis = np.clip(dispL, 0,255)
    vis = np.uint8(ip.rescale(vis, (0,255)))
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imshow('dispL', vis_color)
       
          
    ch = cv2.waitKey(0)    
    cv2.destroyAllWindows()
        