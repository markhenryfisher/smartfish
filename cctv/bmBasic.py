# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:40:06 2018

@author: Mark
"""

import numpy as np
import cv2
import stereoDisparity
import cctv_utils as cctv
import argparse
from noisy import noisy
import os

def matchTemplate(img, template, roi):
    """
    performs template matching
    """
    
    r,c,rr,cc = roi
    h,w = img.shape[:2]
    
    # for loop limits
    mfirstr = r
    mlastr = min(h, r+rr) # clamp mlastr
    mfirstc = c
    mlastc = min(w, c+cc) # clamp mlastc
    
    SAD = np.iinfo('uint16').max
    loc = (mfirstr, mfirstc)
          
    
    for i in range(mfirstr, mlastr):
        mminr = max(0, i-halfBlockSize) 
        mmaxr = min(h, i+halfBlockSize+1)
        for j in range(mfirstc, mlastc):
            mminc = max(0, j-halfBlockSize) 
            mmaxc = min(w, j+halfBlockSize+1)
            windowLeft = img[mminr:mmaxr,mminc:mmaxc]
            mm = np.sum(abs(template - windowLeft))
#            print(i,j,mm)
            if mm < SAD:
                loc = (i,j)
                SAD = mm
                
        
    return loc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute stereo disparity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, default="../data/",
                        help='Root pathname.')
    parser.add_argument('--imgL', type=str, default="beltE56.tif",
                        help='Left image filename.')
    parser.add_argument('--imgR', type=str, default="beltE55.tif",
                        help='Right image filename.')
    parser.add_argument('--dx', type=int, default=8.58,
                        help='Stereo baseline')
    args = parser.parse_args()
    
    # global variables and switches
    r = 0.15
    # Note: use dx-1 to ensure we don't over cook translation and miss!
    dx = np.floor(args.dx)
    
    
    # read raw images
    rawL = cv2.imread(args.root_path+args.imgL)
    rawR = cv2.imread(args.root_path+args.imgR)
    
    imgL = cv2.cvtColor(rawL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(rawR, cv2.COLOR_BGR2GRAY)
    
    # translate the Left image (bring L & R into alighnment)
    imgL = cctv.translateImg(imgL, (-dx, 0))
    
    # generate noise
    noiseImg = noisy("gauss", np.zeros_like(imgL), 0.1) * 100
    
    # match template
    template = cv2.imread(args.root_path+'template.tif',0)
    mask = cctv.matchTemplate(imgR,template)
    
    # preprocess
    imgL = stereoDisparity.preprocessImg(imgL, r)
    imgR = stereoDisparity.preprocessImg(imgR, r)

    # add noise to belt
    imgL[np.where( mask==255)] = noiseImg[np.where( mask==255)]
    imgR[np.where( mask==255)] = noiseImg[np.where( mask==255)]
    
    L = imgL.copy()
    R = imgR.copy()
    
    h,w = L.shape[:2]
    Dbasic = np.zeros((h,w), np.int32)
    disparityRange = 15
    # set block size
    halfBlockSize = 7
    blockSize = 2*halfBlockSize+1
    
    # for loop limits
    firstr = halfBlockSize # first row
    lastr = h-halfBlockSize-1 # last row
    firstc = halfBlockSize # first col
    lastc = w-halfBlockSize-1 # last col
    
    # Scan over all rows.
    for m in range(firstr, lastr):
        # Set min/max row bounds for image block.
        minr = max(0, m-halfBlockSize) 
        maxr = min(h, m+halfBlockSize+1)
        # Scan over all columns
        for n in range(firstc, lastc):
            minc = max(0, n-halfBlockSize) 
            maxc = min(w, n+halfBlockSize+1)
            # compute disparity bounds
#            mind = max(-disparityRange, 1-minc)
            mind = 0
            maxd = min( disparityRange, w-maxc)
            # construct template and region of interest
            template = R[minr:maxr,minc:maxc]
            th, tw = template.shape
            templateCenter = (th//2, tw//2)
            roi = (minr+templateCenter[0], minc+templateCenter[1], 1, maxd-mind)
            # run the template matcher
            loc = matchTemplate(L, template, roi)
#            print(roi[1], loc[1] )
            Dbasic[roi[0], roi[1]] = loc[1] - roi[1]
#        
#    cv2.imshow('preL', fusedL)
#    
    cv2.imshow('L', L)
    cv2.imshow('R', R)
     
    
    vis = np.uint8(cctv.rescale(Dbasic, (0,255))) 
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imshow('Dbasic', vis_color)
    vis = cctv.imfuse(vis_color, rawR, 0.5)
    cv2.imshow('final', vis)
    ch = cv2.waitKey(0)    
    cv2.destroyAllWindows()

        
        