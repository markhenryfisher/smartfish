#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spyder Editor
03.01.19 - removed template from preprocessing step. Changed way disparity is rendered. 
02.01.19 - changes to computeDisparity to give 'normalised' disparity,
            and messed with the colormaps
22.12.18 - changes to processDisparity to calculate many disparity estimates.
20.12.18 - copied to cctv_processVideo
Script to test cctv video input, FrameBuffer, Disparity
@filename: processVideo_v1.py
@author: mark.fisher@uea.ac.uk
@last_updated: 20.12.18
"""
import numpy as np
import cv2
import sys
sys.path.append('C:/Users/Mark/opencv-master/samples/python')
from common import draw_str
import cctv_utils as cctv
import argparse
from statistics import median
import cctv_disparity as cctvDisp
import os
#import matplotlib.image as mpimg


class FrameBuffer:
        def __init__(self, s):
            self.size = s
            self.data = []
            self.x = []
            self.count = 0
            self.comb = self.__disp_comb()
            
        def __disp_comb(self):
            """
            comb - find number of dissparity combinations supported by buffer
            """
            n = int(0)
            for i in range(self.size-1,-1,-1):
                for j in range(i-1,-1,-1):
                    n += 1
                    
            return n
                    
            
        def push(self, f1):
            if len(self.data) < 1:
                x = 0.0
            else:
                f0 = self.data[0]
                x = self.x[0]
                f0Gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                f1Gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                dx = cctv.getBeltMotionByOpticalFlow(f0Gray, f1Gray)
                if len(dx) == 0:
                    x += 0
                else:
                    x += median(dx)
            if len(self.data) < self.size:
                pass
            else:
                raise Exception('FrameBuffer: Buff Full!')            
            self.data.insert(0, f1)
            self.x.insert(0, x)
            self.count += 1
            
        def pop(self):
            self.count -= 1
            return self.data.pop(), self.x.pop()
            
        def nItems(self):
            return self.count

  

def computeDisparity(imgL, imgR, dx):
    """
    computeDisparity - run cctv stereo processing pipeline
    """
#    global template

    imgL, imgR, dx_ = cctvDisp.stereoPreprocess(imgL, imgR, dx)
    dispL, dispR, wlsL, wlsConf = cctvDisp.cctvDisparity(imgL, imgR, dx, alg='sgbm', iFlag=True) 

    # normalise disparity
    out = ( dispL/16.0 + dx_ ) / dx 
     
    return out

    
def processDisparity(buff, count):
    global temp_path
    global debug
    
    imgRef = buff.data[-1]
    h, w = imgRef.shape[:2]
    z_buff = np.zeros((h,w,buff.comb))
    n = 0
    dxMax = buff.x[0] - buff.x[-1]
    
    print("Processing %s frames; Ref frame %s; Belt transport %s." % (buff.nItems(),count,dxMax))    
    for i in range(buff.nItems()-1,-1,-1):
        for j in range(i-1,-1,-1):
            imgR = buff.data[i]
            imgL = buff.data[j]
            # stereo baseline
            dx = buff.x[j] - buff.x[i]
            # reference translation
            tdx = np.int(np.round(buff.x[i] - buff.x[-1]))
            if debug:
                print('imgR= %s : imgL= %s : dx= %s' % (i,j,dx))
            disp = computeDisparity(imgL, imgR, dx)
            disp = cctv.translateImg(disp, (-tdx, 0))
            z_buff[:,:,n] = disp
            n += 1
    
    # compute sum and average disparity
    sumDisp = np.sum(z_buff,axis=2)
    avDisp = sumDisp.copy() / z_buff.shape[2] 
    avDisp = np.uint8(cctv.rescale(avDisp, (0,255), (0.95, 1.25)))
    avDisp[:,-int(dxMax):-1] = 0
    vis_color = cv2.applyColorMap(avDisp, cv2.COLORMAP_JET) 
    vis_mean = cctv.imfuse(imgRef, vis_color, 0.2)
#    vis_mean[:,-int(dxMax):-1] = 0
    frametxt = "Ref frame: %s; Belt dx: %s." % (count,round(dxMax))    
    draw_str(vis_mean, (20, 20), frametxt)
    
#    sumDisp = np.uint8(cctv.rescale(sumDisp, (0,255)))
#    sumDisp[:,-int(dxMax):-1] = 0
#    vis_color = cv2.applyColorMap(sumDisp, cv2.COLORMAP_JET)
#    vis_sum = cctv.imfuse(imgRef, vis_color, 0.2)
#    draw_str(vis_sum, (20, 20), frametxt)
    
    if debug:
        # write results to file
#        filename = temp_path+"Sum"+str(count)+".jpg"
#        cv2.imwrite(filename, vis_sum)
        filename = temp_path+"Mean"+str(count)+".jpg"
        cv2.imwrite(filename, vis_mean)
        
        # display results on screen
#        cv2.imshow('Sum'+str(count), cv2.applyColorMap(sumDisp, cv2.COLORMAP_JET))
        cv2.imshow('Mean'+str(count), cv2.applyColorMap(avDisp, cv2.COLORMAP_JET))
        
        cv2.waitKey(0)
#        cv2.destroyWindow('Sum'+str(count))
        cv2.destroyWindow('Mean'+str(count))
         
    return vis_mean
    
        
def process(cam, params, *args):
    global debug
    buffSize = args[0]
    start = args[1]
    stop = args[2]
    count = 0
    buff = FrameBuffer(buffSize)
    
    for i in range(buffSize):
        _ret, f = cam.read()
        f = cctv.rectify(f, params)
        buff.push(f)
    
    while True:
        f = buff.data[-1]
        x = buff.x[-1]
        
        vis = f.copy()
        draw_str(vis, (20, 20), 'Frame: %d D: %f' %(count,x))
    
        if count>=start:
            cv2.imwrite('../data/beltE' + str(count) + '.tif', f)
            out_frame = processDisparity(buff, count)
            out.write(out_frame)
            cv2.imshow('video', vis)
            cv2.imshow('Disparity', out_frame)
            if debug:
                ch = cv2.waitKey(0)
                if ch == 27:
                    cv2.destroyAllWindows()
                    out.release()
                    break
            else:
                cv2.waitKey(1)
        if count>=stop:
             cv2.destroyAllWindows()
             out.release()
             break
            
        f, x = buff.pop()
        _ret, f = cam.read()
        f = cctv.rectify(f, params)
        buff.push(f)
        count += 1
        
def parse_args():
    parser = argparse.ArgumentParser(description='process video to find stereo disparity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, default="../data/",
                        help='Root pathname.')
    parser.add_argument('--video_file', type=str, default="Belt E base.mp4",
                        help='video filename.')
    parser.add_argument('--template_file', type=str, default="template.tif",
                        help='template filename.')
    parser.add_argument('--cal_file', type=str, default="beltE/cameraParams.yml",
                        help='calibration filename.')
    parser.add_argument('--start', type=int, default=55, help='Start at frame=start_idx')
    parser.add_argument('--stop', type=int, default=200, help='Stop at frame=stop_idx')
    args = parser.parse_args()
    
    return args
        

if __name__ == '__main__':

    args = parse_args()
    buffSize = 5
    
    # debug switch
    global debug
    debug = True
    
    # creates a temporary directory to save data generated at runtime
    global temp_path
    temp_path = args.root_path+'temp/'
    try:
        os.makedirs(temp_path)
    except OSError:
        if os.path.isdir(temp_path):
            pass
    
    # global variables
    cameraParams = cctv.getCameraParams(args.root_path+args.cal_file)
    # template not needed (03.01.19)
#    global template
#    template = cv2.imread(args.root_path+args.template_file,0)
#    template = None
    # for debug
    #global z, vis

    frame_width = 540
    frame_height = 414
    outvidfilename = temp_path+'outpy.avi'
    out = cv2.VideoWriter(outvidfilename,cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))
    
    print('Spooling to frame %s ...' % args.start)
    cam = cv2.VideoCapture(args.root_path+args.video_file)
    process(cam, cameraParams, buffSize, args.start, args.stop)
    
    
    
            