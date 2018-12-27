# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:16:33 2018

@author: Mark

Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""

import numpy as np
#import os
#import cv2

def noisy(noise_typ, image, *args):
    

    
    if np.ndim(image) == 2:
        row,col = image.shape
        ch = 1
        image = image.reshape(row,col,ch)
    else:
        row,col,ch= image.shape    
    
    
    if noise_typ == "gauss":
#        row,col,ch= image.shape
        if len(args)>0:
            var = args[0]
        else:
            var = 0.1
        mean = 0
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy_img_clipped = np.clip(noisy, 0, 255)
        if ch==1:
            noisy_img_clipped = noisy_img_clipped.reshape(row,col)

        return noisy_img_clipped
    elif noise_typ == "s&p":
#        row,col,ch = image.shape
        if len(args)>0:
            amount = args[0]
        else:
            amount = 0.004
        s_vs_p = 0.5
        out = np.copy(image)
        out = out.flatten()
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = np.random.randint(0, out.size - 1, int(num_salt))
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = np.random.randint(0, out.size - 1, int(num_pepper))
        out[coords] = 0
        if ch == 1:
            out = out.reshape(row,col)
        else:
            out = out.reshape(row,col,ch)
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
#        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
    return noisy
