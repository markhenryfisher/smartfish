# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:53:04 2019

@author: Mark
"""

def refineBeltMotionByStructuralSimilarity(f0, f1, dx):
    from utils import image_plotting as ip
    from skimage.measure import compare_ssim as ssim
#    import numpy as np
#    import cv2
    
    sMax = 0
    for delta in range (-2, +2):
        temp = ip.translateImg(f1, (dx+delta, 0))
#        vis = np.uint8(ip.blend(f0, temp, 0.5))
#        cv2.imshow('vis', vis)
#        cv2.waitKey(0)
        s = ssim(f1, temp, multichannel=True)
        if s > sMax:
            sMax = s
            delta_Best = delta
            
    return dx+delta_Best

def cluster(dxdy,k): 
    import numpy as np
    from sklearn.cluster import KMeans
    
    X = np.asarray(dxdy)
    
    kmeans = KMeans(n_clusters=2)  
    kmeans.fit(X)
    
    return kmeans.cluster_centers_
    

def getBeltMotionByOpticalFlow(f0, f1):
    """
    23.01.19 - crop frames to give right-hand-side (i.e. cheat). We addopt this
    strategy because although clustering works we need to track objects over several frames.
    This means propagating cluster lables from one frame to another. We will look at this
    at a later date.
    22.01.19 - now look for two clusters (one due to an arm, one due to the belt;)
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

     
    h,w = f0.shape[:2]
    f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    
    f0 = f0[:, w//2:-1]
    f1 = f1[:, w//2:-1]
    
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.01, #0.3,
                       minDistance = 7,
                       blockSize = 7 )
      
    tracks = []
    good_tracks = []    
    dxdy = []
    
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
        dxdy.append([x0-x1, y0-y1])
            
    centres = cluster(dxdy,2)
    
          
    return centres[:,0]