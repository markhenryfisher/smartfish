# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:47:41 2019
20.01.19 - No longer need to convert to gray as done by belt travel.

@author: Mark
"""
from belt import belt_travel as bt
from statistics import median

class FrameBuffer:
    
    def __init__(self, s):
        self.size = s
        self.data = []
        self.x = []
        self.count = 0
        self.comb = self.__disp_comb()
        self.direction = None
        
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
        """
        get motion check that direction of belt travel is consistant
        """  
        threshold = 2
    
        dx = bt.getBeltMotionByOpticalFlow(f0, f1)
        if len(dx) == 0:
            print('Warning: No Features to Track!!!')
            dx = 0
        else:
            dx = median(dx)
#            print('dx = %f' % dx)
    
        if abs(dx) < threshold:
            pass
        else:
            if dx < 0:
                direction = 'forwards'
            else:
                direction = 'backwards'
                
            if self.direction is None:
                self.direction = direction
            else:
                if direction != self.direction:
                    print('Warning: Tracking Error!!!')
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
