# -*- coding: utf-8 -*-
"""
Spyder Editor

@filename: rotate.py (https://stackoverflow.com/questions/6802577/rotation-of-3d-vector)
@author: mark.fisher@uea.ac.uk
@created: 01.05.19 
"""
import numpy as np
    
def rotate(X, theta, axis='x'):
    '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': return np.dot(X, np.array([
            [1.,  0,  0],
            [0 ,  c, -s],
            [0 ,  s,  c]
            ]))
    elif axis == 'y': return np.dot(X, np.array([
            [c,  0,  -s],
            [0,  1,   0],
            [s,  0,   c]
            ]))
    elif axis == 'z': return np.dot(X, np.array([
            [c, -s,  0 ],
            [s,  c,  0 ],
            [0,  0,  1.],
            ]))   
    
if __name__ == '__main__':
    xyz = np.zeros((2,2,3))
    xyz[0,0,:] = [0,0,0]
    xyz[0,1,:] = [0,1,0]
    xyz[1,0,:] = [1,1,1]
    xyz[1,1,:] = [1,0,0]
    result = rotate(xyz, np.pi/2)
    print(result[:,:,0])
    print(result[:,:,1])
    print(result[:,:,2])    