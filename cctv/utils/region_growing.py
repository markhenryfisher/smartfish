# -*- coding: utf-8 -*-
# !/usr/bin/enc python
"""Provides a simple implementation of region growing
"""
import sys
import numpy as np
import PIL
import utils.myutils as myutils
import matplotlib.pyplot as plt
from scipy import ndimage


__author__ = "Mark Fisher"
__copyright__ = "Copyright 2019"
__licence__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Mark Fisher"
__email__ = "Mark.Fisher@uea.ac.uk"
__status__ = "Dev"

def simple_region_growing(img, seed, threshold=20, mean=0, mask = np.array([])):
    """
    Name: simple_region_growing
    Author: Mark.Fisher@uea.ac.uk
    Purpose: implements region growing image segmentation
    Parameters: 
        img :  2d array (ndarray), Type = uint8
        seed : seed pixel (row,col), Type = tuple
        threshold : region threshold, Type = int
        mean : initial region mean, Type = int
        mask : initial region mask (ndarray), Type = bool 
    Returns:
        reg_image : 2d array (ndarray), Type = bool
        
        Citation: From Matlab by D. Kroon, Univ. of Twente
    """
    try:
        dims = img.shape
    except TypeError:
        raise TypeError("(%s) img : imgplimgmage expected!" % (sys._getframe().f_code.co_name))
        
    if not(len(dims) == 2):
        raise TypeError("(%s) img: 2d array expected!" % (sys._getframe().f_code.co_name))

    # threshold tests
    if (not isinstance(threshold, int)) :
        raise TypeError("(%s) Int expected!" % (sys._getframe().f_code.co_name))
    elif threshold < 0:
        raise ValueError("(%s) Positive value expected!" % (sys._getframe().f_code.co_name))

    if not((isinstance(seed, tuple)) and (len(seed) is 2) ) :
        raise TypeError("(%s) (x, y) variable expected!" % (sys._getframe().f_code.co_name))

    if (seed[0] or seed[1] ) < 0 :
        raise ValueError("(%s) Seed should have positive values!" % (sys._getframe().f_code.co_name))
    elif ((seed[0] > dims[0]) or (seed[1] > dims[1])):
        raise ValueError("(%s) Seed values greater than img size!" % (sys._getframe().f_code.co_name))

    # parameters
    if len(mask)==0:
        reg_img = np.zeros(img.shape) # Output
    else:
        reg_img = np.logical_not(mask).astype(float)*2
        
    pix_area = dims[0]*dims[1]
    
    if mean == 0:    
        reg_mean = float(img[seed]) # The mean of the segmented region
    else:
        reg_mean = float(mean)
        
    reg_size = 1 # Number of pixels in region
    
    neg_free = 10000
    neg_pos = 0
    neg_list = np.zeros((neg_free, 3))
    
    pix_dist = 0 # Distance of the regions newest pixel to the region mean
    
    # TODO: may be enhanced later with 8th connectivity
    orient = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
    cur_pix = seed
    
    #Spreading
    while(pix_dist<threshold and reg_size<pix_area):
    #adding pixels
        for j in range(4):
            #select new candidate
            temp_pix = [cur_pix[0]+orient[j][0], cur_pix[1]+orient[j][1]]
    
            #check if it belongs to the image
            is_in_img = dims[0]>temp_pix[0]>0 and dims[1]>temp_pix[1]>0 #returns boolean
            #candidate is taken if not already selected before
            if (is_in_img and (reg_img[tuple(temp_pix)]==0)):
                neg_list[neg_pos,:] = [temp_pix[0], temp_pix[1], img[tuple(temp_pix)]]
                reg_img[tuple(temp_pix)] = 1
                neg_pos = neg_pos+1
            
        # Add a new block of free memory
        if (neg_pos+10>neg_list.shape[0]):
            neg_list = np.append(neg_list, np.zeros((10000, 3)), axis=0)
            
        # Add pixel with intensity nearest to the mean of the region, to the region
        dist = abs(neg_list[0:neg_pos,2]-reg_mean)
        # return index of the first minima
        idx = np.where(dist == min(dist))[0]
        pix_dist = dist[idx[0]]
#        print('distance: {}'.format(pix_dist))
        reg_img[tuple(cur_pix)] = 2
        reg_size = reg_size+1
        
        # Calculate new mean of the region
        reg_mean = (reg_mean*reg_size+neg_list[idx[0],2])/(reg_size+1)
        
        # Update cur_pix
        cur_pix = [neg_list[idx[0],0].astype(int), neg_list[idx[0],1].astype(int)]
        
        # remove the pixel from the neighbour (check) list
        neg_list = np.delete(neg_list, idx[0], 0)
        neg_pos = neg_pos-1
        
#        vis = np.ubyte(reg_img>1) * 255       
#        cv2.imshow('reg', vis)
#        k = cv2.waitKey(0)
#        if k == 27:
#            break
        
    reg_img = reg_img>1
        
    return np.logical_not(reg_img), reg_size
    
def getseed(img):
    """
    Name: seed
    Author: Mark.Fisher@uea.ac.uk
    Purpose: implements region growing image segmentation
    Parameters: 
        img :  PIL image object
    Returns:
        list of seed (tuple) (row, col)
    """
    # convert to np array
    I = myutils.PIL2array(img)
    # find a good seed pixel (correcting error in matlab code!)
    # Note: seed pixel position (x,y)
#    Er = np.array([I[:,0], I[:,I.shape[1]-1]]).T
#    Ec = np.array([I[0,:], I[I.shape[0]-1,:]])
#    colmin = Ec.min()
#    rowmin = Er.min()
#    if rowmin < colmin:
#        idx = np.where(Er == rowmin)
#        seed = (idx[0][0], idx[1][0] * (I.shape[1]-1))
#    else:
#        idx = np.where(Ec == colmin)
#        seed = (idx[0][0] * (I.shape[0]-1), idx[1][0])  
    
    seed = []
    Er = np.array([I[:,0], I[:,I.shape[1]-1]]).T
    Ec = np.array([I[0,:], I[I.shape[0]-1,:]])
    
    for i in range(2):
        colmin = Ec[i,:].min()
        colidx = np.where(Ec[i,:] == colmin)
        rowmin = Er[:,i].min()
        rowidx = np.where(Er[:,i] == rowmin)
        rowseed = (rowidx[0][0], i * (I.shape[1]-1))
        colseed = (i * (I.shape[0]-1), colidx[0][0])
        seed.append(rowseed)
        seed.append(colseed)
    
    return (seed)
    
def seg_foreground_object(img,threshold=20):
    """
    Name: seg_foreground_object
    Author: Mark.Fisher@uea.ac.uk
    Purpose: implements multi-scale region growing image segmentation
    Parameters: 
        img :  PIL image object
        threshold : region threshold, Type = int
    Returns:
        seg_img
        mask_img
    """
    tWidth = 150
    tHeight = int((float(img.size[1])*float(tWidth/float(img.size[0]))))
    img150 = img.resize((tWidth, tHeight), PIL.Image.ANTIALIAS)
    
#    cv2.imshow('tnail',myutils.PIL2array(img150))
#    cv2.waitKey(0)
    
    # try region growing from a few candidate seed pixels
    reg_size = 0
    seed = getseed(img150)
    for s in seed:    
        temp, sz = simple_region_growing(myutils.PIL2array(img150),s,threshold)
        if sz > reg_size:
           reg = temp.copy()
           reg_size = sz
    
    reg = myutils.filterBoundaryObjects(reg, tWidth)
    
#    vis = np.ubyte(reg) * 255       
#    cv2.imshow('reg', vis)
#    cv2.waitKey(0)
#
#    cv2.destroyAllWindows()
    mask_img = myutils.getBiggestObject(reg)
    
    # 2nd Pass if image is large
    if img.size[0] > tWidth:
        fullWidth = 1024
        wpercent = fullWidth/float(img.size[0])
        fullHeight = int((float(img.size[1])*float(wpercent)))
        img1024 = img.resize((fullWidth,fullHeight), PIL.Image.ANTIALIAS)
        # parameters to grow region again at high resolution
        I = myutils.PIL2array(img1024)
        # M = myutils.imresize(mask_img, I.shape).astype(bool)
        D1 = myutils.imresize(ndimage.binary_dilation(mask_img.astype(float)), I.shape)
        D2 = myutils.imresize(ndimage.binary_dilation(D1.astype(float)), I.shape)
        #N = myutils.imresize(D2, I.shape).astype(bool)
        #mean = np.mean(I[D2==False])
        mean = 0
        DS = np.logical_xor(D1, D2)
        # Find a good seed pixel
        min_idx = np.where(I[np.where(DS==True)] == min(I[np.where(DS==True)]))
        idx = np.where(DS)
        seed = idx[0][min_idx[0][0]], idx[1][min_idx[0][0]]
        
        # grow region (2nd pass)
        temp, __ = simple_region_growing(I,seed,20,mean,D2)
        mask_img = myutils.getBiggestObject(temp).astype(float)
            
    return myutils.array2PIL(mask_img)
    
if __name__ == "__main__":
    print("Testing region growing")
    filename = 'C:/fish/data/fish2018stereo/debug/MRV SCOTIA/frame223.jpg'
#    filename = 'sample.png'
    sample = PIL.Image.open(filename).convert('L')
    sample = myutils.array2PIL(np.uint8(np.abs(np.float64(myutils.PIL2array(sample))-255)))
    # enhance edges
    sample = sample.filter(PIL.ImageFilter.EDGE_ENHANCE)
    
    # segmentation
    mask = seg_foreground_object(sample)
    
    # this line added on 07.04.2019
    mask = myutils.array2PIL(myutils.PIL2array(mask) * 255)
    
    plt.subplot(211)
    plt.imshow(sample, cmap=plt.cm.gray, interpolation='nearest')
    plt.subplot(212)
    plt.imshow(mask, cmap='Greys',  interpolation='nearest')
    
    plt.show()
