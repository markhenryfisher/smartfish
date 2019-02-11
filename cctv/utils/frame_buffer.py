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
    
    def __init__(self, s, d, path):
        self.size = s
        self.raw = []
        self.data = []
        self.x = []
        self.count = 0
        self.comb = self.__disp_comb()
        self.direction = d
        self.template = cv2.imread(path+'template.tif', cv2.IMREAD_GRAYSCALE)
        
    def __disp_comb(self):
        """
        comb - find number of dissparity combinations supported by buffer
        """
        n = int(0)
        for i in range(self.size-1,-1,-1):
            for j in range(i-1,-1,-1):
                n += 1                
        return n
    
    def __check_motion(self,f0,f1):
#        dx = bt.getBeltMotionByOpticalFlow(f0, f1, self.template)
#        
#        dx = np.array(dx)
#        
##        print(dx)
#            
#        if len(dx) == 0:
#            print('Warning: No Features to Track!!!')
#            dx = 0    
#        else:         
#            dx = np.mean(dx)
            
        dx, conf = bt.getBeltMotionByTemplateMatching(f0, f1)
        if conf < 0.9:
            print('Warning: No Confidence in Belt Position!!!')
        
        return dx
                
        
    def push(self, r1, f1):
        if len(self.data) < 1:
            x = 0.0
        else:
            f0 = self.data[0]
            x = self.x[0]
            x += self.__check_motion(f0, f1)
            
        if len(self.data) < self.size:
            self.raw.insert(0, r1)
            self.data.insert(0, f1)
            self.x.insert(0, x)
            self.count += 1
        else:
            raise Exception('FrameBuffer: Buff Full!')            
        
        
    def pop(self):
        self.count -= 1
        return self.raw.pop(), self.data.pop(), self.x.pop()
        
    def nItems(self):
        return self.count
