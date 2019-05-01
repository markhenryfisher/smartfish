# -*- coding: utf-8 -*-
"""
Misc Utils for SGBM Stereo

@author MHF
"""   

def getQmatrix(cam_matrix, dx):
    import numpy as np
    
    fx = cam_matrix[0,0] # focal length
    cx = cam_matrix[0,2] # principal point x-coord
    cy = cam_matrix[1,2] # principal point y-coord
    Tx = dx # stereo baseline length
    
    Q = np.array([[1, 0, 0, -cx], 
                  [0, 1, 0, -cy], # -1 in this row to turn points 180 deg around x-axis,
                  [0, 0, 0, fx],  # so that y-axis looks up
                  [0, 0, 1/Tx, 0]])          
    
    return Q
    
    

def stereoPreprocess(imgL, imgR, alpha = 0.75, k = 3): 
    """
    stereoPreprocess - Prepares images for stereo disparity
    Use Sobel kernel size = 3 for cctv, = 5 for hdtv 
    """  
    from utils import image_plotting as ip
    import numpy as np
    import cv2
    
    outL = imgL.copy()
    outR = imgR.copy()
    
    imgLgrey = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRgrey = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    imgLequ = cv2.equalizeHist(imgLgrey)
    imgRequ = cv2.equalizeHist(imgRgrey)
    
    # prefilter
    edgeL = np.uint8(ip.rescale(abs(cv2.Sobel(imgLequ,cv2.CV_64F,1,0,ksize=k)), (0,255)))
    edgeR = np.uint8(ip.rescale(abs(cv2.Sobel(imgRequ,cv2.CV_64F,1,0,ksize=k)), (0,255)))
    # blend edges and raw
    if imgL.ndim == 1:
        outL = np.uint8(ip.blend(edgeL, imgL, alpha))
    else:
        for idx in range(imgL.ndim):
            outL[:,:,idx] = np.uint8(ip.blend(edgeL, imgL[:,:,idx], alpha))
            
    if imgR.ndim == 1:
        outR = np.uint8(ip.blend(edgeR, imgR, alpha))
    else:
        for idx in range(imgR.ndim):
            outR[:,:,idx] = np.uint8(ip.blend(edgeR, imgR[:,:,idx], alpha))
    
    return outL, outR

def inpaintUnmatchedBlocks(src):
    import numpy as np
    import cv2
    
    m = np.min(src[src > np.min(src)])
    dest = cv2.inpaint(src, (np.uint8(src < m) * 255), 3,cv2.INPAINT_NS)
    
    return dest
   

def getDispRange(dx):
    """
    getDispRange - compute minDisp and numDisp based on dx
    """
    import numpy as np
    
    delta = int(np.floor(dx))
    minDisp = delta - delta % 16
    maxDisp = delta - minDisp
    numDisp = maxDisp + (16 - maxDisp % 16 ) #+ 16
    
    return minDisp, numDisp

def sgbmDisparity(imgL, imgR, params, minDisp=0, numDisp= 16, fFlag=False):
    """
    sgbmDisparity - compute disparity using semi-global block matching
    Note: Set fFlag to filter output with weighted least squares 
    """
    import cv2
     
    __, windowSize, p1, p2 = params
        
    if (numDisp<=0 or numDisp%16!=0):
        raise NameError('Incorrect max_disparity value: it should be positive and divisible by 16')

    if(windowSize<=0 or windowSize%2!=1):
        raise NameError('Incorrect window_size value: it should be positive and odd')
    
    stereoL = cv2.StereoSGBM_create(minDisparity = minDisp,
        numDisparities = numDisp,
        blockSize = windowSize,
        P1 = p1,
        P2 = p2,
        disp12MaxDiff = 1,
#        preFilterCap = 63,
        uniquenessRatio = 20, 
        speckleWindowSize = 64,
        speckleRange = 32,
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
    

def bmDisparity(imgL, imgR, params, minDisp, numDisp):
    import cv2
    
    if imgL.ndim == 3:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    if imgR.ndim == 3:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    __, windowSize, __, __ = params
    
    if (numDisp<=0 or numDisp%16!=0):
        raise NameError('Incorrect max_disparity value: it should be positive and divisible by 16')

    if(windowSize<=0 or windowSize%2!=1):
        raise NameError('Incorrect window_size value: it should be positive and odd')

    stereoL = cv2.StereoBM_create(numDisp,windowSize)
    stereoL.setMinDisparity(minDisp) 
    
    stereoR = cv2.ximgproc.createRightMatcher(stereoL)
    
    dispL = stereoL.compute(imgL, imgR)
    dispR = stereoR.compute(imgL, imgR)
    
    # return raw disparity
    return dispL, dispR

def findDisparity(imgL, imgR, params, minDisp=0, numDisp=16, fFlag=False, iFlag=False):
    """
    cctvDisparity - computes disparity
    """
    import numpy as np
    
    # unpack params
    alg, __, __, __ = params
    wlsL = wlsConf = None


    if alg=='sgbm':
        dispL, dispR, wlsL, wlsConf = sgbmDisparity(imgL, imgR, params, minDisp, numDisp, fFlag=fFlag)
        
    elif alg=='bm':
        dispL, dispR = bmDisparity(imgL, imgR, params, minDisp, numDisp)
    else:
        assert (False), 'Unknown disparity algorithm'
        
    if iFlag:
        dispL = inpaintUnmatchedBlocks(np.float32(dispL))
    else:
        pass
        
    return dispL, dispR, wlsL, wlsConf


def compute3d(imgL, imgR, dx, tdx, params, cam_matrix, iFlag=True, debug=False):
    """
    computeDisparity - run cctv stereo processing pipeline
    """
#    import cv2
    import numpy as np
    from utils import image_plotting as ip
    import cv2

    imgL, imgR = stereoPreprocess(imgL, imgR)
    
    minDisp = 0
    numDisp=16
    
#    cv2.imshow('imgL', imgL)
#    cv2.imshow('imgR', imgR)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
       
    dispL, __, __, __ = findDisparity(ip.translateImg(imgL, (dx, 0)), imgR, params, minDisp=minDisp, numDisp=numDisp)
    
    
    h,w = dispL.shape
    offset = min([minDisp, 0])
    dispL[:,w-int(-dx)+offset:w] = (minDisp-1)*16
    
    # align disparity with reference imgR (set any "new" pixels to unmatched value)
    dispL = ip.translateImg(dispL, (-tdx, 0)) 
    dispL[:,w-tdx:w] = (minDisp-1)*16
    
    # find non-matching disparity values
    no_match_idx = np.where(dispL==(minDisp-1)*16)
     
    # make all pixels +ve
    dispL = dispL - ((minDisp-1)*16)

    
    # recover 3D image
    baseline = abs(dx)
    temp = np.float32((dispL / 16.0) + baseline)
    Qmatrix = getQmatrix(cam_matrix, baseline)
    xyz = cv2.reprojectImageTo3D(temp, Qmatrix) 

    # replace non-matching disparity values with NaN 
    depth = xyz[:,:,2]
    depth[no_match_idx] = np.nan
    xyz[:,:,2] = depth
    
    return xyz

def process_frame_buffer(buff, count, iFlag = True, debug = 0, temp_path = './'):
    """
    find_disparity_from_frame_buffer - compute disparity map for frame pairs
    in frame buffer and fuse into ONE map.
    """
    import numpy as np
    import cv2
    from utils import image_plotting as ip
    from utils import region_growing as rg
    from utils import myutils as myutils
    from utils import my3dtransforms
#    from belt import belt_travel as bt
    import os
    import PIL
    
    
    
    if buff.belt_name == 'MRV SCOTIA': # belt has no texture
        params = ('sgbm', 15, 2*3*15**2, 8*3*15**2)
    else:
        params = ('sgbm', 15, 2*1*15**2, 8*1*15**2)
        
    camera_matrix = buff.video.lens_calibration.camera_matrix
    threshold = buff.minViableStereoBaseline * 3 / 4
    imgRef = buff.data[-1].copy()
    h, w = imgRef.shape[:2]
    # save successsive depth maps to depthStack
    depthStack = []
    # observation window 
    watch = []
    n = 0
    max_maps = 8
#    dxMax = buff.getLargestStereoBaseline()
    #print("\nProcessing %s frames; Ref frame %s; Belt transport %s." % (buff.nItems(),count,dxMax))    
    for i in range(buff.nItems-1,-1,-1):
        for j in range(i-1,-1,-1):
            imgR = buff.data[i].copy()
            imgL = buff.data[j].copy()
            # stereo baseline
            dx = buff.x[i] - buff.x[j]
            
            # recheck stereo baseline
#            new_dx = bt.getBeltMotionByTemplateMatching(buff.belt_name, imgL, imgR, max_travel=int(abs(dx)+5))
#            dx = new_dx[0]
                       
            # we assume belt moves left-to-right
            if buff.direction == 'backwards':
                imgR = np.flip(imgR,axis=1)
                imgL = np.flip(imgL,axis=1)
                dx = -dx
                         
            # reference translation
            tdx = abs(np.int(np.round(buff.x[-1] - buff.x[i])))

            frameR_n = count+(buff.nItems-1)-i
            frameL_n = count+(buff.nItems-1)-j
            if abs(dx)>threshold and n < max_maps: # todo fix this
                if debug > 0:
                    print('imgR= {} : imgL= {} : dx= {:.2f}'.format(frameR_n,frameL_n,dx))                                
             
                xyz = compute3d(imgL, imgR, dx, tdx, params, camera_matrix, iFlag, debug)
                watch.append(xyz[140:145,w-485:w-480,2])
                depthStack.append(xyz[:,:,2])      
                n += 1
    
    # post process depthStack        
    depthArr = np.asarray(depthStack)
    if n>0:
        # average disparity maps ignoring nan entries
        avDepth = np.nanmean(depthArr, axis=0) 
        # map nan values onto a depth plane 
        avDepth[np.isnan(avDepth)] = camera_matrix[0,0] #-np.inf
    else:
        avDepth = np.zeros((h,w), dtype=np.float64)
    
    # flip image (undo earlier flip)    
    if buff.direction == 'backwards':
        avDepth = np.fliplr(avDepth)
        for i in range(depthArr.shape[0]):
            depthArr[i,:,:] = np.fliplr(depthArr[i,:,:])

    #  load (or create) roi mask 
    if os.path.isfile(os.path.join(temp_path, "mask"+str(count)+".npy")):
        mask = np.load(os.path.join(temp_path, "mask"+str(count)+".npy"))
    else:
        img_grey = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
        img_bright_foreground = myutils.array2PIL(np.uint8(np.abs(np.int16(img_grey)-255)))
        # enhance edges
        img_bright_foreground = img_bright_foreground.filter(PIL.ImageFilter.EDGE_ENHANCE)
        mask = myutils.PIL2array(rg.seg_foreground_object(img_bright_foreground))
        mask = cv2.resize(mask, (w,h), cv2.INTER_NEAREST) > 0
        np.save(os.path.join(temp_path, "mask"+str(count)), mask)
           
      
    # make 3d array of x,y,z coords
    xyz[:,:,2] = avDepth               
    # force depth == focal length outside mask (disparity == baseline)
    xyz[np.logical_not(mask),2] = camera_matrix[0,0]
    
    
    # find ROI
    im2, ctr, hiers  = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(ctr[0])


    # correct for camera pose
    # rotate pi rads around x-axis so z axis points up
    xyz = my3dtransforms.rotate(xyz, np.pi)

    #save the 3d point cloud; examine depth samples     
    if debug > 0:
        colors = cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB)
        # rotate pi radians around x axis so z axis points up
        roi_points = xyz[y:y+h,x:x+w,:]
        roi_colors = colors[y:y+h,x:x+w,:]
        # examine depth samples
        filename = os.path.join(temp_path, "plot"+str(count)+".png")
        ip.plot_transept((x,y,w,h), depthArr, imgRef, filename)
        
        # write to file
        filename = os.path.join(temp_path, "xyz"+str(count)+".ply")
        ip.write_ply(filename, roi_points, roi_colors)
        
    visRef = imgRef.copy()
    cv2.rectangle(visRef, (x,y), (x+w-2,y+h-2), (0,255,0), 2)
    

    # rotate -pi/4 around y-axis so camera is orthogonal to belt (a guess)
#    xyz = my3dtransforms.rotate(xyz, np.pi/16, axis='y')
    # force depth == focal length outside mask (disparity == baseline)
#    xyz[np.logical_not(mask),2] = -camera_matrix[0,0]
    
    # set depth == focal length outside roi (disparity == baseline)
#    avDepth[np.logical_not(mask)] = camera_matrix[0,0]
    
    # render out1, out2
    visDepth = np.uint8(ip.rescale(xyz[:,:,2], (0,255)))    
    out1 = cv2.applyColorMap(visDepth, cv2.COLORMAP_JET) 
    out2 = ip.overlay(imgRef, visDepth)
    
    if debug > 0:
        # write results to file
        filename = os.path.join(temp_path, "frame"+str(count)+".jpg") 
        cv2.imwrite(filename, imgRef)
        filename = os.path.join(temp_path, str(n)+"mean"+str(count)+".jpg")
        cv2.imwrite(filename, out1)
        filename = os.path.join(temp_path, "map"+str(count)+".png")
        ip.show_depth_with_scale(xyz[:,:,2],filename)
         
    return visRef, out1, out2