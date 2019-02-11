#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains some modules used by cctv programs

Filename: cctv_utils.py
Date: 07.10.2018
Author mark.fisher@uea.ac.uk
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def translateImg(img, delta):
    """
    translateImg - shift image by delta (dx, dy)
    """
    
    dx, dy = delta
    rows,cols = img.shape[:2]

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
       
    return dst

def matchTemplate(image, template, threshold=0.1):
    """
    matchTemplate - return mask showing areas of image that match template
                    using match metric cv2.TM_SQDIFF_NORMED 
    """
    assert ( len(image.shape) <= 2), "Image is not greyscale"
    assert ( len(template.shape) <= 2), "Template is not greyscale"

    mask = np.zeros_like(image, dtype=np.uint8)
    w,h = template.shape[::-1]
    # invoke template Matching
    res = cv2.matchTemplate(image,template,cv2.TM_SQDIFF_NORMED)
    loc = np.where( res <= threshold)
    for pt in zip(*loc[::-1]):
         mask[pt[1]:pt[1]+h+1, pt[0]:pt[0]+w+1] = 255
    
    return mask
    

def imfuse(imgX, imgY, x):
    """
    imfuse - mix two images in proportion x, 1-x
    """    
    X = np.float64(imgX.copy())
    Y = np.float64(imgY.copy())
    
    Z = np.uint8((X * x) + (Y * (1.0-x)))
    
    return Z

def plotTransept(src, row, filename):
    fig = plt.figure()
    plt.plot(src[row,:])
    plt.ylabel('Raw Disparity')
    plt.show()
    
    if filename is not None:
        fig.savefig(filename, dpi=fig.dpi)

def cap(img, bounds):
    lower, upper = bounds
    img[img<lower] = lower
    img[img>upper] = upper


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
        np.clip(src, m, M)
    else:
        m = np.min(x)
        M = np.max(x)
    
    if M-m < np.finfo(np.float64).eps:
        y = x
    else:
        y = (b-a) * (x-m)/(M-m) + a
        
    return y



def getBeltMotionByOpticalFlow(f0, f1):
    """
    19.01.19 - changed param maxLevel 2 -> 3
    10.01.19 - now returns +ve or -ve dx vals (not abs())
    getBeltMotionByOpticalFlow(f0, f1) - find fisheries CCTV belt motion
    input:
        f0, f1 - consecutive video frames
    output:
        dx - estimate of belt travel (pixels)
    """   
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
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

        
def pointsToWorld(pts, rvec, tvec, cameraMatrix):
    """
    pointsToWorld(pts, rvec, tvec, cameraMatrix) - transforms image points to world points
    (implements MATLAB pointsToWorld)
    input:
        pts - vector of 4 image point coords (4, 1, 2)
        rvec - rotation vector
        tvec - translation vector
        cameraMatrix - camera intrinsic matrix
    output:
        Y - vector of transformed points (4, 1, 2)
    """
    pts = pts.reshape(-1,2)
    R, _ = cv2.Rodrigues(rvec)
    t1 = np.array([R[:, 0], R[:, 1], tvec])
    tform = t1.dot(cameraMatrix.T)   
    X = np.concatenate([pts, np.ones((pts.shape[0],1), dtype=pts.dtype)], axis=1) 
    invTform = np.linalg.inv(tform)
    U = np.dot(X, invTform)
    
    if U.size == 0:
        Y = np.zeros_like(pts)
    else:
        U[:,0] = U[:,0] / U[:,2]
        U[:,1] = U[:,1] / U[:,2]
        Y = U[:,0:2]
        
    return np.float32(Y.reshape(4,1,2))

def drawGrid(vis, sqSz):
    """
    drawGrid(vis, sqSz) - draws square grid on image
    input:
        vis - image
        sqSz - size of grid
    output:
        vis - image with grid inserted
    """
    h, w = vis.shape[:2]
    # draw horizontals
    for i in range (sqSz, h-1, sqSz):
        cv2.line(vis, (0, i), (w-1, i), (0, 255, 0))
    # draw verticals
    for i in range (sqSz, w-1, sqSz):
        cv2.line(vis, (i, 0), (i, h-1), (0, 255, 0)) 
    
    return vis

def getCameraParams(cal_file):
    """
    getCameraParams(cal_file) - reads camaraParams from calibration file
    input:
        cal-file - calibration file (yml) produced by calibrate.py
    output:
        cameraParams - data structure (see https://stackoverflow.com/questions/35988/c-like-structures-in-python)
    """   
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    
    fs = cv2.FileStorage(cal_file, cv2.FILE_STORAGE_READ)
    K = fs.getNode('camera_matrix').mat()
    D = fs.getNode('distortion_coefficients').mat()
    tvecs = fs.getNode('extrinsicT').mat()
    rvecs = fs.getNode('extrinsicR').mat()
    beltROI = fs.getNode('beltROI').mat()
    cv2.FileStorage.release(fs)
    
    cameraParams = Bunch(intrinsicMatrix=K, 
                         distortionCoefficients=D,
                         rotationVectors = rvecs,
                         translationVectors = tvecs,
                         beltROI = beltROI)
    
    return cameraParams
    
def rectify(img, camParams):
    """
    rectify(img, camParams) - Performs image rectification for fisheries cctv
    input:
        img - raw cctv camera image frame (640 x 480)
        camParams - cameraParams struct (see getCameraParams( ) for detail)
    output:
        imgr - rectified image (croped to beltROI)
    """ 
    camera_matrix = camParams.intrinsicMatrix
    dist_coeffs = camParams.distortionCoefficients
    tvecs = camParams.translationVectors
    rvecs = camParams.rotationVectors
    beltROI = camParams.beltROI 
    
    # choose i in range 0-3 (it doesn't make much difference)
    i = 0
    # get size of beltROI in world coords, choice of beltROI[i] arbitray
    # Note: point coords (x,y) or (y,x) depending on extrinsics 
    (x0,y0,_),(x1,y1,_),(x2,y2,_),(x3,y3,_) = beltROI[i] 
    if abs(y0-y1) > 0.1:
        pass
    else:
        (y0,x0,_),(y1,x1,_),(y2,x2,_),(y3,x3,_) = beltROI[i]
       
    beltW, beltH = abs(y0-y1), abs(x0-x3)         
    pts2 = np.array([[0,0], [beltW-1, 0], [beltW-1, beltH-1], [0, beltH-1]], dtype = np.float32)  

    # project beltROI
    pts1, _ = cv2.projectPoints(np.float32(beltROI[i]), rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

    # undistort the image
    imgu = cv2.undistort(img, camera_matrix, dist_coeffs)
    # undistort the (projected) beltROI
    pts1u = cv2.undistortPoints(pts1, camera_matrix, dist_coeffs, None, camera_matrix)
 
    # perform perspective (rectification) transform
    tformR = cv2.getPerspectiveTransform(pts1u, pts2)
    imgR = cv2.warpPerspective(imgu, tformR, (int(beltW), int(beltH)))
    
    return imgR

