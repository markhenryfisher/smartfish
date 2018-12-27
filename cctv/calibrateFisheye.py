#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibrate fisheries CCTV camera using fisheye camera model
Created on Sun Oct 28 18:31:11 2018

@author: mark.fisher@uea.ac.uk
@filename: calibrateFisheye.py
@lastupdated: 01.11.2018
"""

import numpy as np
import cv2
import argparse
import os
from glob import glob

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='camera calibration for beltE distorted images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_mask', type=str, default="../data/beltE/frame-*.tif",
                        help='Directory containing calibration files.')
    args = parser.parse_args()
    
    img_mask = args.img_mask
    img_names = glob(img_mask)
    
    cal_mask = os.path.dirname(img_mask) + '/frame*.yml'
    cal_names = glob(cal_mask)
    fs = cv2.FileStorage(cal_names[0], cv2.FILE_STORAGE_READ)
    board_width = int(fs.getNode('board_width').real())
    board_height = int(fs.getNode('board_height').real())
    square_size = fs.getNode('square_size').real()
    h = int(fs.getNode('image_height').real())
    w = int(fs.getNode('image_width').real())
    cv2.FileStorage.release(fs)
    
    # 'M' indicates object points consistent with matlab calibrate
    version = 'M'
#    intrinsic_guess = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], np.float64)
    # Note: pattern_size = (rows, cols) matches order of points given in calibration files      
    pattern_size = (board_height, board_width)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    if version == 'M':
        pattern_points[:, :2] = np.fliplr(np.indices(pattern_size).T.reshape(-1, 2))
    else:
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    
    obj_points = []
    img_points = []
    
    for filename in cal_names:
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        corners = fs.getNode('corners').mat()
        # Note: reshape required to fix error thrown by cv2.fisheye.calibrate ( )
        img_points.append(corners.reshape(1,corners.shape[0],corners.shape[1]))
        obj_points.append(pattern_points.reshape(1,pattern_points.shape[0],pattern_points.shape[1]))
    
    cv2.FileStorage.release(fs)

    # calculate fisheye camera distortion
    cv2.__version__[0]
    
    N_OK = len(obj_points)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

#Transposing both object and image points to shapes (1, <num points in set> , 3)
# and (1, <num points in set> , 2), respectively, seems like a workaround for the above issue.

#    obj_points = np.asarray(obj_points)
#    obj_points = obj_points.reshape(1, -1, 3)
#    img_points = np.asarray(img_points)
#    img_points = img_points.reshape(1, -1, 2)
    
#    obj_points = np.float32(obj_points)
#    img_points = np.float32(img_points)

    rms, K, D, rvecs, tvecs = \
        cv2.fisheye.calibrate(
            obj_points,
            img_points,
            (w, h),
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        
    print('fisheye camera calibration: ')
    print("RMS:", rms)   
    print("camera matrix:\n", K)
    print("distortion coefficients: ", D.ravel())
    print('\n')
    
    
    # compute reprojection errors using camera matrix
    print("Reprojection errors from camera_matrix:")
    tot_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.fisheye.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
        imgpoints2 = np.float32(imgpoints2).reshape(-1,2)
        error = cv2.norm(img_points[i].reshape(-1,2),imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        print("average reprojection error, pattern", i, ":", error)
        tot_error += error

    print("average reprojection error: ", tot_error/len(obj_points))
    print('\n')
    
    
    # undistort the images with the calibration
    for img_found in img_names:
        img = cv2.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        
        dst = cv2.fisheye.undistortImage(img, K, D=D, Knew=K.copy())
               
        cv2.imshow('raw', img)
        cv2.imshow('corrected', dst)

        print('Press any key...')
        ch = cv2.waitKey(0)

    # rectangular belt ROI in world coords (i.e. chess board squares), 
    # coord values chosen by trial and error
    # [[start, top, 0], [end, top, 0], [end, bottom, 0], [start, bottom, 0]]        
    beltROI = np.array([[[[5.5, -2, 0], [5.5, 15.5, 0], [-5.5, 15.5, 0], [-5.5, -2, 0]]],
                        [[[-2.3, 10, 0], [-2.3, -6, 0], [8, -6, 0], [8, 10, 0]]],
                        [[[-9, -1.8, 0], [6.6, -1.8, 0], [6.6, 7.7, 0],  [-9, 7.7, 0]]],
                        [[[14.5, 6.9, 0], [-1, 6.9, 0], [-1, -2.7, 0], [14.5, -2.7, 0]]]
                        ], dtype = np.float32)
    beltROI *= square_size
    
#    imgpoints2, _ = cv2.fisheye.projectPoints(beltROI[3], rvecs[3], tvecs[3], K, D)
#    imgpoints2 = np.float32(imgpoints2).reshape(-1,2)
#    vis = img.copy()
#    cv2.circle(vis, tuple(imgpoints2[0]), 5, (255, 0, 0))
#    cv2.imshow('out0', vis)
#    
#    cv2.circle(vis, tuple(imgpoints2[1]), 5, (255, 0, 0))
#    cv2.imshow('out1', vis)
#    
#    cv2.circle(vis, tuple(imgpoints2[2]), 5, (255, 0, 0))
#    cv2.imshow('out2', vis)
#    
#    cv2.circle(vis, tuple(imgpoints2[3]), 5, (255, 0, 0))
#    cv2.imshow('out3', vis)
#    
#    ch = cv2.waitKey(0)
    
    
    # write camera parameters to file
    filename = os.path.dirname(img_mask) + '/fisheyeParams.yml'
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    cv2.FileStorage.write(fs, 'version', version)
    cv2.FileStorage.write(fs, 'nframes', len(img_names))
    cv2.FileStorage.write(fs, 'board_width', board_width)
    cv2.FileStorage.write(fs, 'board_height', board_height)
    cv2.FileStorage.write(fs, 'square_size', square_size)
    cv2.FileStorage.write(fs, 'aspect_ratio', 1.0)
    cv2.FileStorage.write(fs, 'camera_matrix', K)
    cv2.FileStorage.write(fs, 'distortion_coefficients', D)
    cv2.FileStorage.write(fs, 'avg_reprojection_error', tot_error/len(obj_points))
    cv2.FileStorage.write(fs, 'extrinsicT', np.array(tvecs))
    cv2.FileStorage.write(fs, 'extrinsicR', np.array(rvecs))
    cv2.FileStorage.write(fs, 'beltROI', beltROI)
    cv2.FileStorage.release(fs)
    
    cv2.destroyAllWindows()
    
    