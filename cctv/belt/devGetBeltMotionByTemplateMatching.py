# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:11:07 2019

@author: Mark
"""
import cv2

def getBeltMotionByTemplateMatching(img0, img1, max_travel=50):
    """
    09.02.19
    getBeltMotionByTemplateMatching() - find fisheries CCTV belt motion
    input:
        img0, img1 - consecutive video frames
        max_travel - max belt travel (default = +/- 50 pixels)
    output:
        dx - estimate of belt travel (pixels)
    """
    import cv2
    
    x,y = img0.shape[:2]
    f0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    f1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    startx = max_travel
    stopx = x-max_travel
    template = f0[:,startx:stopx]
    
    # Apply template Matching
    res = cv2.matchTemplate(f1,template,cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    dx = min_loc[0] - max_travel
    confidence = 1 - min_val
    
    return dx, confidence


if __name__ == '__main__':
    """
    Test stub for getBeltMotionByTemplateMatching
    """ 
    
    # read images
    img0 = cv2.imread('../../data/beltE55.tif')
    img1 = cv2.imread('../../data/beltE57.tif')
    
    x, conf = getBeltMotionByTemplateMatching(img0, img1)
    
    print('Belt Travel = %s; Confidence = %.2f' % (x, conf))