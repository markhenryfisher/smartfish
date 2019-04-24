# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:47:41 2019
20.01.19 - changed median -> mean;
20.01.19 - added __check_motion method 
20.01.19 - No longer need to convert to gray as done by belt travel.

@author: Mark
"""
from belt import belt_travel as bt
import numpy as np
import cv2


class FrameBuffer:
    vga_shape = (480, 640)
    hd_shape = (800, 1280)
    
    def __init__(self, s, d, video):
        self.size = s
        self.raw = []
        self.data = []
        self.x = []
        self.count = 0
        self.comb = self.__disp_comb()
        self.direction = d
        self.belt_name = video.belt.name
        self.minViableStereoBaseline = self.__viable_stereo_baseline(video.img_shape)
        self.video = video
        
    def __reset(self, r, f):
        """
        reset the buffer
        """
        self.raw = []
        self.data = []
        self.x = []
        self.count = 0
        self.push(r,f)
        
        
    def __disp_comb(self):
        """
        comb - find number of dissparity combinations supported by buffer
        """
        n = int(0)
        for i in range(self.size-1,-1,-1):
            for j in range(i-1,-1,-1):
                n += 1                
        return n
    
    def __viable_stereo_baseline(self,shape):
        """
        baseline - set value according to img_size
        """
        if  shape == self.vga_shape:
            x = 70.0
        elif shape == self.hd_shape:
            x = 70.0
        else:
            raise RuntimeError('Unknown video resolution')
            
        return x
    
    def __check_motion(self,f0,f1):
        err = False    
        dx, conf = bt.getBeltMotionByTemplateMatching(self.belt_name, f0, f1)
        if conf < 0.9:
            err = True
        if (self.direction == 'forwards' and dx < 0) or (self.direction == 'backwards' and dx > 0):
            err = True

        return dx, err
                
        
    def push(self, r1, f1):
        if len(self.data) < 1:
            x = 0.0
            err = False
        else:
            f0 = self.data[0]
            x = self.x[0]
            dx, err = self.__check_motion(f0, f1)
            x += dx
        
        if err:
            print('Belt transport tracking Error...resetting buffer')
            self.__reset(r1,f1)
        else:
            if len(self.data) < self.size:
                self.raw.insert(0, r1)
                self.data.insert(0, f1)
                self.x.insert(0, x)
                self.count += 1
            else:
                raise RuntimeError('FrameBuffer: Buff Full!')
        
    
    def getLargestStereoBaseline(self):
        return abs(self.x[0] -  self.x[-1])
    
    def getLastdx(self):
        return self.x[-2] - self.x[-1]
                
    def pop(self):
        self.count -= 1
        return self.raw.pop(), self.data.pop(), self.x.pop()
        
    def nItems(self):
        return self.count
    
    @property
    def dxAverage(self):
        if self.count < 2:
            dst = 0.0
        else:
            dst = self.x[0] / float(self.count - 1)
            
        return dst
    
    @property
    def sufficientStereoBaseline(self):
        return self.getLargestStereoBaseline() > self.minViableStereoBaseline
