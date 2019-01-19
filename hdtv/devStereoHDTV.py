# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:28:35 2019

@filename: devStereoHDTV
@author: mark.fisher@uea.ac.uk
"""

import cv2
import numpy as np
from statistics import median
from utils import image_plotting as ip
import argparse

def getBeltMotionByOpticalFlow(f0, f1):
    """
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
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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
        if abs(y0-y1) < 1:
            dx.append(x0-x1)
       
    return dx


def rescale(src, bounds, *args):
    """
    rescale - rescale data in (a,b)
    y = rescale(x,bounds, minMax)
    Note: minMax is an optional argument that sets m and M; this allows scalefactor 
    to be set independently.
    
    """
    x = np.float64(src.copy())
    a, b = bounds
    if len(args)>0:
        m, M = args[0]
        cap(x, (m, M))
    else:
        m = np.min(x)
        M = np.max(x)
    
    if M-m < np.finfo(np.float64).eps:
        y = x
    else:
        y = (b-a) * (x-m)/(M-m) + a
        
    return y


def translateImg(img, delta):
    """
    translateImg - shift image by delta (dx, dy)
    """
    
    dx, dy = delta
    rows,cols = img.shape[:2]

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
       
    return dst

def stereoPreprocess(imgL, imgR, dx, alpha = 0.25):
    """
    stereoPreprocess - Prepares images for stereo disparity
    """   
    # make dx a conservative estimate so we should always find a match
    dx = int(dx-dx/50.0)  
    # translate
    imgL = translateImg(imgL, (dx, 0))
    # prefilter
    edgeL = np.uint8(np.clip(cv2.Sobel(imgL,cv2.CV_64F,1,0,ksize=3), 0,63))
    edgeR = np.uint8(np.clip(cv2.Sobel(imgR,cv2.CV_64F,1,0,ksize=3), 0,63))
    # blend edges and raw
    imgL = np.uint8(ip.blend(edgeL, imgL, alpha))
    imgR = np.uint8(ip.blend(edgeR, imgR, alpha))
    
    return imgL, imgR, dx

def inpaintUnmatchedBlocks(src):
    m = np.min(src[src > np.min(src)])
    dest = cv2.inpaint(src, (np.uint8(src < m) * 255), 3,cv2.INPAINT_NS)
    
    return dest
   
def getDispRange(dx):
    """
    getDispRange - compute minDisp and numDisp based on dx
    """
    delta = int(np.floor(dx))
    minDisp = delta - delta % 16
    maxDisp = delta - minDisp
    numDisp = maxDisp + (16 - maxDisp % 16 ) # + 16
    
    return minDisp, numDisp

def sgbmDisparity(imgL, imgR, dx=0, fFlag=False):
    """
    sgbmDisparity - compute disparity using semi-global block matching
    Note: Set fFlag to filter output with weighted least squares 
    """
     
    windowSize = 15
    
    minDisp, numDisp = getDispRange(dx)

 
    print('sgbm disparity, dx= ', dx)
    print('minDisp: %s, numDisp: %s, windowSize: %s' % (minDisp,numDisp, windowSize))
    
    if (numDisp<=0 or numDisp%16!=0):
        raise NameError('Incorrect max_disparity value: it should be positive and divisible by 16')

    if(windowSize<=0 or windowSize%2!=1):
        raise NameError('Incorrect window_size value: it should be positive and odd')
    
    stereoL = cv2.StereoSGBM_create(minDisparity = minDisp,
        numDisparities = numDisp,
        blockSize = windowSize,
        P1 = 8*1*windowSize**2,
        P2 = 32*1*windowSize**2,
        disp12MaxDiff = 1,
#        preFilterCap = 63,
        uniquenessRatio = 5,
#        speckleWindowSize = 25,
#        speckleRange = 5,
        mode = cv2.STEREO_SGBM_MODE_HH
    )    
    stereoR = cv2.ximgproc.createRightMatcher(stereoL)
    
    dispL = stereoL.compute(imgL, imgR)
    dispR = stereoR.compute(imgR, imgL)
    
    
    # fiter the output
    if fFlag:
        # FILTER Parameters
        lmbda = 80000
        sigma = 0.8 #1.2
        
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoL)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        wlsL = wls_filter.filter(dispL, imgL, None, dispR)  # important to put "imgL" here!!!
        wlsConf = wls_filter.getConfidenceMap( )
    else:
        wlsL = wlsConf = None
        
    
    # return raw disparity
    return dispL, dispR, wlsL, wlsConf
    

def bmDisparity(imgL, imgR, dx=0):
    
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    windowSize = 15
    
    minDisp, numDisp = getDispRange(dx)
    
    print('bm disparity, dx= ', dx)
    print('numDisp', numDisp)
    
    if (numDisp<=0 or numDisp%16!=0):
        raise NameError('Incorrect max_disparity value: it should be positive and divisible by 16')

    if(windowSize<=0 or windowSize%2!=1):
        raise NameError('Incorrect window_size value: it should be positive and odd')

    stereoL = cv2.StereoBM_create(numDisp,windowSize)
    stereoL.setMinDisparity(minDisp)

    
#    stereoL.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
#    stereoL.setPreFilterCap(7)
#    stereoL.setPreFilterSize(15)
#    stereoL.setDisp12MaxDiff(7)
#    stereoL.setSpeckleWindowSize(50)
#    stereoL.setSpeckleRange(5)
#    
    
    stereoR = cv2.ximgproc.createRightMatcher(stereoL)
    
    dispL = stereoL.compute(imgL, imgR)
    dispR = stereoR.compute(imgL, imgR)
    
    # return raw disparity
    return dispL, dispR

def beltDisparity(imgL, imgR, dx=0, alg='sgbm', fFlag=False, iFlag=False):
    """
    beltDisparity - computes disparity
    """
    
    wlsL = wlsConf = None

    if alg=='sgbm':
        dispL, dispR, wlsL, wlsConf = sgbmDisparity(imgL, imgR, fFlag=fFlag)
        
    elif alg=='bm':
        dispL, dispR = bmDisparity(imgL, imgR)
    else:
        assert (False), 'Unknown disparity algorithm'
        
    if iFlag:
        dispL = inpaintUnmatchedBlocks(np.float32(dispL))
           
    return dispL, dispR, wlsL, wlsConf

def parse_args():
    parser = argparse.ArgumentParser(description='process video to find stereo disparity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, default="../data/belt_images/SUMMER DAWN PD97/",
                        help='Root pathname.')
    parser.add_argument('--f0name', type=str, default="frame_110.tif",
                        help='frame0 image filename.')
    parser.add_argument('--f1name', type=str, default="frame_111.tif",
                        help='frame1 image filename.')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    
    # read frames
    f0 = cv2.imread(args.root_path+args.f0name)
    f1 = cv2.imread(args.root_path+args.f1name)
    
    # ensure belt motion is left-to-right
    rawR = np.flip(f0,axis=1)
    rawL = np.flip(f1,axis=1)
    
    dx = getBeltMotionByOpticalFlow(rawR, rawL)
    if len(dx) == 0:
        raise ValueError('Warning: Tracker failed!!!')
    dx = median(dx)
     
    imgL, imgR, dx_ = stereoPreprocess(rawL, rawR, dx)
#    
#        
    cv2.imshow('rawL', rawL)
    cv2.imshow('rawR', rawR)
    cv2.imshow('imgL', imgL)
    cv2.imshow('imgR', imgR)
    
    #  compute disparity          
    dispL, dispR, wlsL, wlsConf = beltDisparity(imgL, imgR, dx=0, alg='sgbm', iFlag=False)

    # display dispaity
    vis = np.clip(dispL, 0,255)
    vis = np.uint8(rescale(vis, (0,255)))
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imshow('dispL', vis_color)
    
    # display fused result
    fused = np.uint8(ip.blend(rawR, vis_color, 0.5))
    cv2.imshow('fused', fused)
    
    ch = cv2.waitKey(0)    
    cv2.destroyAllWindows()