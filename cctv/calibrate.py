#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera calibration for fisheries CCTV distorted images with positions of 
calibration matrix points predefined and stored in files (defaults set for beltE).
Camera parameters are written to file '../data/beltE/calibration/cameraParams.yml'

Note: version controls compatability with MATLAB code. 'M' forces obj_points
to be same as MATLAB code.

usage:
    calibraye.py [<imageMask>]

default values:
    <image mask> defaults to ../data/beltE/calibration/frame*.png

Created on Fri Oct 12 15:56:24 2018
@filename: calibrate.py
@author: Mark Fisher@uea.ac.uk
@last_updated: 28.10.18 - added criteria; no longer saving image_width and 
image_height. After adjusting criteria I decided to stick to default.
"""
import numpy as np
import cv2 as cv
import argparse
import os
from glob import glob

calibration_params = dict( flags = cv.CALIB_USE_INTRINSIC_GUESS,
                          criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                          )



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
    fs = cv.FileStorage(cal_names[0], cv.FILE_STORAGE_READ)
    board_width = int(fs.getNode('board_width').real())
    board_height = int(fs.getNode('board_height').real())
    square_size = fs.getNode('square_size').real()
    h = int(fs.getNode('image_height').real())
    w = int(fs.getNode('image_width').real())
    cv.FileStorage.release(fs)
    
    # 'M' indicates object points consistent with matlab calibrate
    version = 'M'
    intrinsic_guess = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], np.float64)
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
        fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
        corners = fs.getNode('corners').mat()
        img_points.append(corners)
        obj_points.append(pattern_points)
    
    cv.FileStorage.release(fs)
    
            
    
    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(objectPoints=obj_points, 
                                                                      imagePoints=img_points, 
                                                                      imageSize=(w, h), 
                                                                      cameraMatrix=intrinsic_guess,
                                                                      distCoeffs=None,
                                                                      **calibration_params)

    print('camera calibration: ')
    print("RMS:", rms)   
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    print('\n')
        
    # compute reprojection errors using camera matrix
    print("Reprojection errors from camera_matrix:")
    tot_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
        imgpoints2 = np.float32(imgpoints2).reshape(-1,2)
        error = cv.norm(img_points[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
        print("average reprojection error, pattern", i, ":", error)
        tot_error += error

    print("average reprojection error: ", tot_error/len(obj_points))
    print('\n')

    
#    # compute reprojecction errors using optimised camera matrix
#    newcameramtx, validPixROI = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
#    print("optimised camera matrix:\n", newcameramtx)
#    print("Reprojection errors from optimised camera_matrix:")
#    tot_error = 0
#    for i in range(len(obj_points)):
#        imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], newcameramtx, dist_coefs)
#        imgpoints2 = np.float32(imgpoints2).reshape(-1,2)
#        error = cv.norm(img_points[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
#        print("average reprojection error, pattern", i, ":", error)
#        tot_error += error
#
#    print("average reprojection error: ", tot_error/len(obj_points))
#    print('\n')
    
    
    # undistort the images with the calibration
    for img_found in img_names:
        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        
        dst = cv.undistort(img, camera_matrix, dist_coefs)
        
##        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
#        newdst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
#
#        # crop and display images
#        x, y, w, h = validPixROI
#        cropped = newdst[y:y+h, x:x+w]
        
        cv.imshow('raw', img)
        cv.imshow('corrected', dst)
#        cv.imshow('optimised', newdst)
#        cv.imshow('cropped', cropped)
        
        print('Press any key...')
        ch = cv.waitKey(0)

    # rectangular belt ROI in world coords (i.e. chess board squares), 
    # coord values chosen by trial and error
    # [[start, top, 0], [end, top, 0], [end, bottom, 0], [start, bottom, 0]]        
    beltROI = np.array([[[5.5, -1, 0], [5.5, 14, 0], [-6, 14, 0], [-6, -1, 0]],
                        [[-2.7, 9, 0], [-2.7, -6, 0], [8.7, -6, 0], [8.7, 9, 0]],
                        [[-9, -2.3, 0], [6.6, -2.3, 0], [6.6, 9, 0],  [-9, 9, 0]],
                        [[14.5, 7.6, 0], [-1, 7.6, 0], [-1, -3.7, 0], [14.5, -3.7, 0]]
                        ], dtype = np.float)
    beltROI *= square_size
        
    # write camera parameters to file
    filename = os.path.dirname(img_mask) + '/cameraParams.yml'
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    cv.FileStorage.write(fs, 'version', version)
    cv.FileStorage.write(fs, 'nframes', len(img_names))
    cv.FileStorage.write(fs, 'board_width', board_width)
    cv.FileStorage.write(fs, 'board_height', board_height)
    cv.FileStorage.write(fs, 'square_size', square_size)
    cv.FileStorage.write(fs, 'aspect_ratio', 1.0)
    cv.FileStorage.write(fs, 'camera_matrix', camera_matrix)
    cv.FileStorage.write(fs, 'distortion_coefficients', dist_coefs)
    cv.FileStorage.write(fs, 'avg_reprojection_error', tot_error/len(obj_points))
    cv.FileStorage.write(fs, 'extrinsicT', np.array(tvecs))
    cv.FileStorage.write(fs, 'extrinsicR', np.array(rvecs))
    cv.FileStorage.write(fs, 'beltROI', beltROI)
    cv.FileStorage.release(fs)
    
    cv.destroyAllWindows()
    
    