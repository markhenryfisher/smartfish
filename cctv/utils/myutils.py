# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:19:38 2015

@author: mhf
"""

# http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/

import numpy as np
from scipy import ndimage
import PIL

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])
    
def array2PIL(img):
    return PIL.Image.fromarray(img.astype('uint8'))

def filterBoundaryObjects(img, threshold):
    """
    Let n = number of object pixels that lie on the image border 
    if n > thershold then remove object  
    """
    im, number_of_objects = ndimage.label(img)
    
    h,w = img.shape
    out_img = img.copy()
    
    
    for i in range(number_of_objects):
        idx = np.where(im == i+1)
        top = np.where(idx[0] == 0)[0]
        left = np.where(idx[1] == 0)[0]
        bottom = np.where(idx[0] == h-1)[0]
        right = np.where(idx[1] == w-1)[0]
        border_pix = top.size + left.size + bottom.size + right.size 
        
        if border_pix > threshold:
            out_img[idx] = False
                    
    return out_img
                    
    
def getBiggestObject(img):
    im, number_of_objects = ndimage.label(img)

    area = np.zeros((number_of_objects,), dtype=np.int)

    for i in range(number_of_objects):
        idx = np.where(im == i+1)[0]
        area[i] = idx.size

    idx = np.where(area == max(area))[0]  

    out_img = im == (idx[0]+1)
    
    return out_img

# resize for nparray    
def imresize(img, size):
    return PIL2array(array2PIL(img).resize((size[1],size[0]), PIL.Image.NEAREST))

# resize for PIL
def PILresize(img, width):
    wpercent = width/float(img.size[0])
    height = int((float(img.size[1])*float(wpercent)))

    return img.resize((width, height), PIL.Image.ANTIALIAS)
    
def applymask(img, mask):
    i = PIL2array(img)
    m = PIL2array(mask).astype(np.bool)
    out = np.zeros(i.shape)
    out[m] = i[m]
    
    return array2PIL(out)
        
    

    