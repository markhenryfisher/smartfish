# -*- coding: utf-8 -*-
"""
Spyder Editor

cctvDisparity( ) - compute disparity
02.01.19 - stereoPreprocess() returns dx (so we can add it to dispL);
            and disabled template matching. dx changed to floor(dx)
01.01.19 - Set sgbm params P1 and P2.
19.12.18 - Tidy up and rename. Much of the main code now done as preprocessing.
19.12.18 - Copied to devStereoDisparity.py
04.12.18 - Add preprocessImg funtion; update and tune sgbm parameters
01.12.18 - Add preprocessing
16.11.18 - Add inpainting
15.11.18 - Revised output display. Added weighted least squares filter.
07.11.18 - Scale output by dividing by dx. Tune minDisp and numDisp.
@filename: disparity_v0.py
@author: mark.fisher@uea.ac.uk
@last_updated: 18.11.18
"""
import numpy as np
import cv2
import argparse
import cctv_utils as cctv
import os
from cctv_noisy import noisy

def stereoPreprocess(imgL, imgR, dx, mix=0.25, template=None):
    """
    stereoPreprocess - Implements all preprocessing steps for beltE
    """ 
    # make dx a conservative estimate so we should always find a match
    dx = np.floor(dx-dx/50.0)
      
    # translate
    imgL = cctv.translateImg(imgL, (-dx, 0))
    
    # prefilter
#    edgeL = np.uint8(cctv.rescale(cv2.Sobel(imgL,cv2.CV_64F,1,0,ksize=3), (0,255)))
#    edgeR = np.uint8(cctv.rescale(cv2.Sobel(imgR,cv2.CV_64F,1,0,ksize=3), (0,255)))
    edgeL = np.uint8(np.clip(cv2.Sobel(imgL,cv2.CV_64F,1,0,ksize=3), 0,63))
    edgeR = np.uint8(np.clip(cv2.Sobel(imgR,cv2.CV_64F,1,0,ksize=3), 0,63))

    # mix edges and raw
    imgL = cctv.imfuse(imgL, edgeL, mix)
    imgR = cctv.imfuse(imgR, edgeR, mix)
    
    # add noise to belt
    if template is not None:
        # match the template
        imgRgrey = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        mask = cctv.matchTemplate(imgRgrey,template,threshold=0.05)
        # generate noise
        noiseImg = noisy("gauss", np.zeros_like(imgRgrey), 0.1) * 100
        loc = np.where( mask==255)
        for pt in zip(*loc[::]):
            imgL[pt[0],pt[1],:] = noiseImg[pt[0],pt[1]]
            imgR[pt[0],pt[1],:] = noiseImg[pt[0],pt[1]]
       
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

 
#    print('sgbm disparity, dx= ', dx)
#    print('minDisp: %s, numDisp: %s, windowSize: %s' % (minDisp,numDisp, windowSize))
    
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

def cctvDisparity(imgL, imgR, dx=0, alg='sgbm', fFlag=False, iFlag=False):
    """
    cctvDisparity - computes disparity
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
    
    parser = argparse.ArgumentParser(description='compute stereo disparity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, default="../data/",
                        help='Root pathname.')
    parser.add_argument('--nameL', type=str, default="beltE61.tif",
                        help='Left image filename.')
    parser.add_argument('--nameR', type=str, default="beltE55.tif",
                        help='Right image filename.')
    parser.add_argument('--dx', type=int, default=132,
                        help='Stereo baseline')
    args = parser.parse_args()
    
    return args
    
    
if __name__ == '__main__':
    """
    Test stub for cctvDisparity
    """    
    args = parse_args()
    
    # beltE dx values wrt x(55) = 0
    # dx(56)=8.58, dx(57)=35.0 , dx(58)=59.68, dx(59)=83.33
    # dx(60)=108, dx(61)=132, dx(62)=155, dx(63)=180, dx(64)=205
    
    # beltE dx values wrt x(56) = 0
    # dx(57)=26.42, dx(58)=51.1 , dx(59)=74.75
    
    # beltE dx values wrt x(57) = 0
    # dx(58)=24.68, dx(59)=48.33
    
    # beltE dx values wrt x(65)=0
    # dx(66)=25.4, dx(67)=51.1, dx(68)=75.1, dx(69)=99.9, dx(70)=125.5, 
    # dx(71)=149.3, dx(72)=173.46, dx(73)=199.157, dx(74)= 223.21
    
    # beltE dx values wrt x(76)=0
    # dx(77)= 24.46, dx(78)= 48.047, dx(79)= 50.56, dx(80)= 50.56
    
    # beltE dx values wrt x(77)=0
    # dx(78)=23.58
    
    # beltE dx values wrt x(78)=0
    # dx(79)=
    
    # beltE dx values wrt x(79)=0
    # dx(80)= 0
    
     
    # creates a temporary directory to save data generated at runtime
    temp_path = args.root_path+'temp/'
    try:
        os.makedirs(temp_path)
    except OSError:
        if os.path.isdir(temp_path):
            pass
        
    # read images
    rawL = cv2.imread(args.root_path+args.nameL)
    rawR = cv2.imread(args.root_path+args.nameR)
#    template = cv2.imread(args.root_path+'template.tif',0)
    
    dx = args.dx
        
    imgL, imgR, __ = stereoPreprocess(rawL, rawR, dx, mix=0.25, template=None)
        
    cv2.imshow('rawL', rawL)
    cv2.imshow('rawR', rawR)
    cv2.imshow('imgL', imgL)
    cv2.imshow('imgR', imgR)
    
    #  compute disparity          
    dispL, dispR, wlsL, wlsConf = cctvDisparity(imgL, imgR, dx=0, alg='sgbm', iFlag=False)
 
    # display dispaity
    vis = np.clip(dispL, 0,255)
    vis = np.uint8(cctv.rescale(vis, (0,255)))
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imshow('dispL', vis_color)
       
    # display fused result
    fused = cctv.imfuse(rawR, vis_color, 0.5)
    cv2.imshow('fused', fused)
          
    ch = cv2.waitKey(0)    
    cv2.destroyAllWindows()
        