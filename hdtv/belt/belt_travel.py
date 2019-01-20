# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:53:04 2019

@author: Mark
"""

def getBeltMotionByOpticalFlow(f0, f1):
    """
    20.01.19 - maxLevel increased from 2->4; 
    17.01.19 - now a standalone function
    10.01.19 - now returns +ve or -ve dx vals (not abs())
    getBeltMotionByOpticalFlow(f0, f1) - find fisheries CCTV belt motion
    input:
        f0, f1 - consecutive video frames
    output:
        dx - estimate of belt travel (pixels)
    """    
    import cv2
    import numpy as np
    
    f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


    feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
      
    tracks = []
    good_tracks = []    
    dx = []
    
    mask = np.zeros_like(f0)
    mask[:] = 255
    p = cv2.goodFeaturesToTrack(f0, mask = mask, **feature_params)
    temp1 = np.float32(p).reshape(-1, 2)
    if p is not None:
        for x, y in temp1:
            tracks.append([(x, y)])
    else:
        raise Exception('No good Features to Track')
        
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    # track forwards f0 -> f1
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(f0, f1, p0, None, **lk_params)
    # track backwards f1 -> f0
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(f1, f0, p1, None, **lk_params)
    # get good tracks
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
        if not good_flag:
            continue
        tr.append((x, y))
        good_tracks.append(tr)
        
    for pt0, pt1 in good_tracks:
        x0, y0 = pt0
        x1, y1 = pt1
        if abs(y0-y1) < 1.0:
            dx.append(x0-x1)
       
    return dx