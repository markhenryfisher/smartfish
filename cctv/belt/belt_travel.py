# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:53:04 2019

@author: Mark
"""

def match_template(img, template, threshold=0.8):
    """
    template_match - find areas of img that match template
    """
    import cv2
    import numpy as np
       
#    # work with greyscale images
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#    method = cv2.TM_SQDIFF_NORMED
#    method = cv2.TM_CCORR_NORMED
    method = cv2.TM_CCOEFF_NORMED
    mask = np.zeros_like(img)
    
    w,h = template.shape[::-1]
    res = cv2.matchTemplate(img,template,method)
    
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        mask[pt[1]:pt[1]+h+1, pt[0]:pt[0]+w+1] = 255
    
    return mask

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
    
    print(kmeans.cluster_centers_)
    
    print(kmeans.labels_)
    
    return kmeans.cluster_centers_

def getBeltMotionByTemplateMatching(img0, img1, max_travel=50):
    """
    13.02.19 - fixed bug.
    09.02.19
    getBeltMotionByTemplateMatching() - find fisheries CCTV belt motion
    input:
        img0, img1 - consecutive video frames
        max_travel - max belt travel (default = +/- 50 pixels)
    output:
        dx - estimate of belt travel (pixels)
        confidence - confidence in estimate (0.0 - 1.0) 
    """
    import cv2
    
    y,x = img0.shape[:2]
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
    

def getBeltMotionByOpticalFlow(img0, img1, template=None):
    """
    29.01.19 - Get lots of errors from belt. Invert mask to select other tracking features.
    29.01.19 - tuned parameters for hdtv footage. Added 'mask' to identify belt.
    replaced clustering by 'mode' (in frame buffer)
    23.01.19 - crop frames to give right-hand-side (i.e. cheat). We addopt this
    strategy because although clustering works we need to track objects over several frames.
    This means propagating cluster lables from one frame to another. We will look at this
    at a later date.
    22.01.19 - now look for two clusters (one due to an arm, one due to the belt;)
    17.01.19 - now a standalone function
    10.01.19 - now returns +ve or -ve dx vals (not abs())
    getBeltMotionByOpticalFlow(f0, f1) - find fisheries CCTV belt motion
    input:
        img0, img1 - consecutive video frames
        template - optional subimage of belt surface texture (used to locate belt regions)
    output:
        dx - list of estimates of belt travel (pixels)
    """    
    import cv2
    import numpy as np
    from utils import image_plotting as ip

     
    h,w = img0.shape[:2]
    f0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    f1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    
#    f0 = f0[:, w//2:-1]
#    f1 = f1[:, w//2:-1]
    
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#                  flags = cv2.OPTFLOW_USE_INITIAL_FLOW)
#                  minEigThreshold = 1e-4 )


    feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 15,
                       blockSize = 7 )
      
    tracks = []
    good_tracks = []    
    dx = []
    
    mask = np.zeros_like(f0)
    if template is None:
        mask[:] = 255
    else:
        mask = match_template(f0, template, 0.5)
        # invert mask (avoid the belt!)
        mask = np.uint8(abs(np.int32(mask)-255)) 
        
    vis = np.uint8(ip.blend(mask,f0,0.5))    
    cv2.imshow('mask', vis)
    cv2.waitKey(1)

    p = cv2.goodFeaturesToTrack(f0, mask = mask, **feature_params)
    temp1 = np.float32(p).reshape(-1, 2)
    if p is not None:
        for x, y in temp1:
            tracks.append([(x, y)])
    else:
        raise Exception('No good Features to Track')
        
#    vis = img0.copy()     
#     # make list of point tuples [(x0, y0), (x1, y1), ...]
#    temp2 = [tr[-1] for tr in tracks]
#    for x, y in temp2:
#        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)    
#    cv2.imshow('good featuures: f0', vis)
#    cv2.imshow('f1', img1)
#    
#    cv2.waitKey(0)
        
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
        if abs(y0-y1) < 1:
            dx.append(x0-x1)
       
    return dx
            
 