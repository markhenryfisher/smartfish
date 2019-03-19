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
    
    Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, fx], [0, 0, 1/Tx, 0]])
    
    return Q
    
    

def stereoPreprocess(imgL, imgR, alpha = 0.25, k = 3): 
    """
    stereoPreprocess - Prepares images for stereo disparity
    Use Sobel kernel size = 3 for cctv, = 5 for hdtv 
    """  
    from utils import image_plotting as ip
    import numpy as np
    import cv2
    
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    imgL = cv2.equalizeHist(imgL)
    imgR = cv2.equalizeHist(imgR)
    
    # prefilter
    edgeL = np.uint8(ip.rescale(abs(cv2.Sobel(imgL,cv2.CV_64F,1,0,ksize=k)), (0,255)))
    edgeR = np.uint8(ip.rescale(abs(cv2.Sobel(imgR,cv2.CV_64F,1,0,ksize=k)), (0,255)))
    # blend edges and raw
    imgL = np.uint8(ip.blend(edgeL, imgL, alpha))
    imgR = np.uint8(ip.blend(edgeR, imgR, alpha))
    
    return imgL, imgR

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
        uniquenessRatio = 50, 
        speckleWindowSize = 256,
        speckleRange = 64,
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
    

def bmDisparity(imgL, imgR, dx=0):
    import cv2
    
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    windowSize = 15
    
    minDisp, numDisp = getDispRange(dx)
    
    print('bm disparity, dx= ', dx)
    print('numDisp', numDisp)
    
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
        dispL, dispR = bmDisparity(imgL, imgR)
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
    
    minDisp = -1
    numDisp=16
#    dispL, __, __, __ = findDisparity(ip.translateImg(imgL, (dx, 0)), imgR, minDisp, numDisp, alg='sgbm', iFlag=iFlag)
       
    dispL, __, __, __ = findDisparity(ip.translateImg(imgL, (dx, 0)), imgR, params, minDisp=minDisp, numDisp=numDisp)
    
    h,w = dispL.shape
    offset = min([minDisp, 0])
    dispL[:,w-int(-dx)+offset:w] = minDisp*16
    
    
    # set unmatched pixels to minDisp
    dispL[np.where( dispL == ((minDisp-1)*16))] = minDisp*16
    # make all pixels +ve
    dispL = dispL - (minDisp*16)
    # translate disparity to compensate for belt motion
    dispL = ip.translateImg(dispL, (-tdx, 0))  
    
    # recover 3D image
    baseline = abs(dx)
    temp = np.float32((dispL / 16.0) + baseline)
    xyz = np.empty([h, w, 3], np.float64)
    Qmatrix = getQmatrix(cam_matrix, baseline)
    xyz = cv2.reprojectImageTo3D(temp, Qmatrix) 
    
    return xyz

def process_frame_buffer(buff, count, iFlag = True, debug = False, temp_path = './'):
    """
    find_disparity_from_frame_buffer - compute disparity map for frame pairs
    in frame buffer and fuse into ONE map.
    """
    import numpy as np
    import cv2
    from utils import image_plotting as ip
    from belt import belt_travel as bt
    import os
    
    
    if buff.belt_name == 'MRV SCOTIA': # belt has no texture
        params = ('sgbm', 15, 8*3*15**2, 32*3*15**2)
    else:
        params = ('sgbm', 15, 2*1*15**2, 8*1*15**2)
        
    camera_matrix = buff.video.lens_calibration.camera_matrix
    threshold = buff.minViableStereoBaseline / (buff.size + 1)
    imgRef = buff.data[-1]
    h, w = imgRef.shape[:2]
    sumDepth = np.zeros((h,w))
    n = 0
    dxMax = buff.getLargestStereoBaseline()
    #print("\nProcessing %s frames; Ref frame %s; Belt transport %s." % (buff.nItems(),count,dxMax))    
    for i in range(buff.nItems()-1,-1,-1):
        for j in range(i-1,-1,-1):
            imgR = buff.data[i]
            imgL = buff.data[j]
            # stereo baseline
            dx = buff.x[i] - buff.x[j]
            
            # recheck stereo baseline
            new_dx = bt.getBeltMotionByTemplateMatching(buff.belt_name, imgL, imgR, max_travel=int(abs(dx)+5))
            dx = new_dx[0]
                       
            # we assume belt moves left-to-right
            if buff.direction == 'backwards':
                imgR = np.flip(imgR,axis=1)
                imgL = np.flip(imgL,axis=1)
                dx = -dx
                         
            # reference translation
            tdx = abs(np.int(np.round(buff.x[-1] - buff.x[i])))

            if debug:
                print('imgR= %s : imgL= %s : dx= %s' % (i,j,dx))
                filename = os.path.join(temp_path, "imgR"+str(count)+".jpg")
                cv2.imwrite(filename, imgR)
                filename = os.path.join(temp_path, "imgL"+str(count)+".jpg")
                cv2.imwrite(filename, imgL)                                
            if abs(dx)>threshold: 
                xyz = compute3d(imgL, imgR, dx, tdx, params, camera_matrix, iFlag, debug)
                sumDepth = sumDepth + xyz[:,:,2]     
                n += 1
                
    # average disparity
    if n>0:
        avDepth = sumDepth / n
    else:
        avDepth = sumDepth
        
    #save the 3d point cloud     
    if debug:
        xyz[:,:,2] = avDepth
        colors = cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB)
        
        #  filter points
        mask = avDepth <= camera_matrix[0,0]
        out_points = xyz[mask]
        out_colors = colors[mask]
        
        # write to file
        filename = os.path.join(temp_path, "xyz"+str(count)+".ply")
        ip.write_ply(filename, out_points, out_colors)
    
    #rescaling set empirically: cctv = (0.02, 0.25) hdtv = (0, 0.12)
    visDepth = np.uint8(ip.rescale(avDepth, (0,255)))
#    avDisp[:,-int(dxMax):-1] = 0
    if buff.direction == 'backwards':
        visDepth = np.fliplr(visDepth)
    
    out1 = cv2.applyColorMap(visDepth, cv2.COLORMAP_JET)   
    out2 = ip.overlay(imgRef, visDepth)
    
    if debug:
        # write results to file
        filename = os.path.join(temp_path, "Mean"+str(count)+".jpg")
        cv2.imwrite(filename, out1)
         
    return out1, out2