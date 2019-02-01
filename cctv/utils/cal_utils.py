# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:30:33 2019

@author: Mark
"""
import cv2
import numpy as np

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
