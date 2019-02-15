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
    parser.add_argument('--frameL', type=int, default=62,
                        help='Left image filename.')
    parser.add_argument('--frameR', type=int, default=56,
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
    right_filename = args.root_path+'beltE'+str(args.frameR)+'.tif'
    left_filename = args.root_path+'beltE'+str(args.frameL)+'.tif'
    print('Right = '+ right_filename+' Left = '+ left_filename)
    rawR = cv2.imread(right_filename)
    rawL = cv2.imread(left_filename)  
    
    # find stereo baseline
    baseline = 0
    for i in range(args.frameR, args.frameL):
      imgR = cv2.imread(args.root_path+'beltE'+str(i)+'.tif')
      imgL = cv2.imread(args.root_path+'beltE'+str(i+1)+'.tif')
      dx, __ = bt.getBeltMotionByTemplateMatching(imgR, imgL) 
      baseline += dx
      
    print('Stereo Baseline = {0:0.2f}'.format(baseline))  


    
    imgL, imgR = stereo_utils.stereoPreprocess(rawL, rawR)
        
    cv2.imshow('rawL', rawL)
    cv2.imshow('rawR', rawR)
    cv2.imshow('imgL', imgL)
    cv2.imshow('imgR', imgR)
    
    #  compute disparity          
    dispL, __, __, __ = stereo_utils.findDisparity(ip.translateImg(imgL, (-baseline, 0)), imgR, minDisp=minDisp, numDisp=numDisp)
 
    h,w = dispL.shape
    offset = min([minDisp, 0])
    dispL[:,w-int(baseline)+offset:w] = minDisp*16
    p = proportion_of_matched_pixels(dispL, minDisp, numDisp)
    print('Fraction of Matched Pixels = {0:0.2f}'.format(p))
  
    #set unmatched pixels to min disparity
    dispL[np.where(dispL == (minDisp-1)*16)] = (minDisp*16)
    # make disparity +ve
    dispL = dispL - (minDisp*16)
    # print some stats
    print('Max= {0:3d}; Min= {1:3d}; Mean= {2:3.2f}.'.format(np.max(dispL), np.min(dispL), np.mean(dispL)))
    
    # display dispaity
    vis = np.uint8(dispL)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imshow('dispL', vis_color)
       
          
    ch = cv2.waitKey(0)    
    cv2.destroyAllWindows()
    
    filename = temp_path+"dispL"+".jpg"
    cv2.imwrite(filename, dispL)
        