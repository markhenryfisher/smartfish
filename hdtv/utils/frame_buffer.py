# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:47:41 2019
20.01.19 - changed median -> mean;
20.01.19 - added __check_motion method 
20.01.19 - No longer need to convert to gray as done by belt travel.

@author: Mark
"""
from belt import belt_travel as bt
#from statistics import median
import numpy as np


class FrameBuffer:
    
    def __init__(self, s, d):
        self.size = s
        self.data = []
        self.x = []
        self.count = 0
        self.comb = self.__disp_comb()
        self.direction = d
        
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
        threshold = 2
    
        dx = bt.getBeltMotionByOpticalFlow(f0, f1)
        
        
        if len(dx) == 0:
            print('Warning: No Features to Track!!!')
            dx = 0
        elif self.direction == 'backwards':
            dx = np.max(dx)
            
        else:
            dx = np.min(dx)
                
        if abs(dx) < threshold:
            dx = 0
    
        return dx
                
        
    def push(self, f1):
        if len(self.data) < 1:
            x = 0.0
        else:
            f0 = self.data[0]
            x = self.x[0]
            x += self.__check_motion(f0, f1)
            
        if len(self.data) < self.size:
            self.data.insert(0, f1)
            self.x.insert(0, x)
            self.count += 1
        else:
            raise Exception('FrameBuffer: Buff Full!')            
        
        
    def pop(self):
        self.count -= 1
        return self.data.pop(), self.x.pop()
        
    def nItems(self):
        return self.count
