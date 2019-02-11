# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def ground_truth(img, dx, template):
    """
    find belt mask
    """
    import cv2
    from utils import image_plotting as ip
    from belt import belt_travel as bt
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ip.translateImg(img, (dx, 0))
    mask = bt.match_template(img, template)
    
#    cv2.imshow('img', img)
#    cv2.imshow('temp', template)
#    cv2.imshow('mask', mask)
#    cv2.waitKey(0)
    
    return mask

#def tweek(imgL, imgR, dx):
#    from utils import image_plotting as ip
#    from skimage.measure import compare_ssim as ssim
#    
#    delta_Best = 0
#    sMax = 0
#    for delta in range (-5, +5):
#        tempL = ip.translateImg(imgL, (dx+delta, 0))
#        s = ssim(imgR, tempL, multichannel=True)
#        if s > sMax:
#            sMax = s
#            delta_Best = delta
#        
#    return dx+delta_Best
        

def stereoPreprocess(imgL, imgR, dx, alpha = 0.25, k = 3):
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
    
    # make dx a conservative estimate so we should always find a match  
#    dx = tweek(imgL, imgR, int(dx))
#    dx = dx
#    print('Tweeked dx = %s' % dx)
    # translate
    #imgL = ip.translateImg(imgL, (dx, 0))
    # prefilter
    edgeL = np.uint8(np.clip(abs(cv2.Sobel(imgL,cv2.CV_64F,1,0,ksize=k)), 0,255))
    edgeR = np.uint8(np.clip(abs(cv2.Sobel(imgR,cv2.CV_64F,1,0,ksize=k)), 0,255))
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

def sgbmDisparity(imgL, imgR, minDisp=0, numDisp= 16, fFlag=False, tFlag=False):
    """
    sgbmDisparity - compute disparity using semi-global block matching
    Note: Set fFlag to filter output with weighted least squares 
    """
    import cv2
     
    windowSize = 15
    
#    if tFlag:
#        minDisp = -7
#        numDisp = 16
#    else:
#        minDisp, numDisp = getDispRange(dx)

 
#    print('sgbm disparity, dx= ', dx)
#    print('minDisp: %s, numDisp: %s, windowSize: %s' % (minDisp,numDisp, windowSize))
    
    if (numDisp<=0 or numDisp%16!=0):
        raise NameError('Incorrect max_disparity value: it should be positive and divisible by 16')

    if(windowSize<=0 or windowSize%2!=1):
        raise NameError('Incorrect window_size value: it should be positive and odd')
    
    stereoL = cv2.StereoSGBM_create(minDisparity = minDisp,
        numDisparities = numDisp,
        blockSize = windowSize,
        P1 = 8*1*windowSize**2,
        P2 = 32*1*windowSize**2,
        disp12MaxDiff = 5,
#        preFilterCap = 63,
        uniquenessRatio = 5,
#        speckleWindowSize = 25,
#        speckleRange = 20,
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

    
#    stereoL.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
#    stereoL.setPreFilterCap(7)
#    stereoL.setPreFilterSize(15)
#    stereoL.setDisp12MaxDiff(7)
#    stereoL.setSpeckleWindowSize(50)
#    stereoL.setSpeckleRange(5)
#    
    
    stereoR = cv2.ximgproc.createRightMatcher(stereoL)
    
    dispL = stereoL.compute(imgL, imgR)
    dispR = stereoR.compute(imgL, imgR)
    
    # return raw disparity
    return dispL, dispR

def findDisparity(imgL, imgR, minDisp=0, numDisp=16, alg='sgbm', fFlag=False, iFlag=False):
    """
    cctvDisparity - computes disparity
    """
    import numpy as np
    
    wlsL = wlsConf = None

    if alg=='sgbm':
        dispL, dispR, wlsL, wlsConf = sgbmDisparity(imgL, imgR, minDisp, numDisp, fFlag=fFlag)
        
    elif alg=='bm':
        dispL, dispR = bmDisparity(imgL, imgR)
    else:
        assert (False), 'Unknown disparity algorithm'
        
    if iFlag:
        dispL = inpaintUnmatchedBlocks(np.float32(dispL))
    else:
#        dispL = np.clip(dispL, 0, 255)
        pass
        
    
        
    return dispL, dispR, wlsL, wlsConf


def computeDisparity(imgL, imgR, dx, template, iFlag=True, debug=False):
    """
    computeDisparity - run cctv stereo processing pipeline
    """
    import cv2
    import numpy as np
    from utils import image_plotting as ip
    
    
    mask = ground_truth(imgL, dx, template)


    imgL, imgR = stereoPreprocess(imgL, imgR, dx)
    
#    if debug:
#        cv2.imshow('preprocessedL', imgL)
#        cv2.imshow('preprocessedR', imgR)
#        cv2.waitKey(0)

#    # 1st pass (find disparity of the belt):
#    minDisp=-7
#    numDisp=16
#    testL, __, __, __ = findDisparity(ip.translateImg(imgL, (dx, 0)), imgR, minDisp, numDisp, alg='sgbm')
#    ground0 = np.zeros_like(mask)
#    loc = np.where( testL > ((minDisp-1)*16))
#    ground0[loc] = 255
#    loc = np.where( mask == 0)
#    ground0[loc] = 0
#
##    cv2.imshow('mask', mask)   
##    cv2.imshow('ground0', ground0)  
##    cv2.waitKey(0)    
#
#    # apply a correction
#    gd0_dx = np.mean(testL[ground0>0]) / 16.0 
#    gd0_sd = np.std(testL[ground0>0]) / 16.0
#    adj = gd0_dx + gd0_sd
#    print('stats: %s %s %s' % (gd0_dx, gd0_sd, adj))
##    minDisp = - int(round(adj))
    minDisp = -1
    numDisp=16
    dispL, __, __, __ = findDisparity(ip.translateImg(imgL, (dx, 0)), imgR, minDisp, numDisp, alg='sgbm', iFlag=iFlag)
       
        
    # set unmatched pixels to minDisp
    dispL[np.where( dispL == ((minDisp-1)*16))] = minDisp*16
    # make all pixels +ve
    dispL = dispL - (minDisp*16)

    # normalise disparity
    #out = ( dispL/16.0 + dx_ ) / dx 
    weight_factor = 1 / (abs(dx) + 1)
    out = ( dispL / 16.0 ) * weight_factor
     
    return out

def process_frame_buffer(buff, count, iFlag = True, debug = False, temp_path = './'):
    """
    find_disparity_from_frame_buffer - compute disparity map for frame pairs
    in frame buffer and fuse into ONE map.
    """
    import numpy as np
    import cv2
    from utils import image_plotting as ip
    import sys
    sys.path.append('C:/Users/Mark/opencv-master/samples/python')
    from common import draw_str
    
    threshold = 20
    imgRef = buff.data[-1]
    h, w = imgRef.shape[:2]
    #z_buff = np.zeros((h,w,buff.comb))
    sumDisp = np.zeros((h,w))
    n = 0
    dxMax = buff.getLargestStereoBaseline()
    print("\nProcessing %s frames; Ref frame %s; Belt transport %s." % (buff.nItems(),count,dxMax))    
    for i in range(buff.nItems()-1,-1,-1):
        for j in range(i-1,-1,-1):
            imgR = buff.data[i]
            imgL = buff.data[j]
            # stereo baseline
##            dx = buff.x[j] - buff.x[i]
            dx = buff.x[i] - buff.x[j]
            
            # we assume belt moves left-to-right
            if buff.direction == 'backwards':
                imgR = np.flip(imgR,axis=1)
                imgL = np.flip(imgL,axis=1)
                dx = -dx
            
            # incremental motion
            #this_dx = abs(buff.x[j] - buff.x[j+1])
            # reference translation
##            tdx = abs(np.int(np.round(buff.x[i] - buff.x[-1])))
            tdx = abs(np.int(np.round(buff.x[-1] - buff.x[i])))
#            dx = abs(dx) 
            if debug:
                print('imgR= %s : imgL= %s : dx= %s' % (i,j,dx))
                #print('this_dx= %s' % this_dx )
                filename = temp_path+"imgR"+str(count)+".jpg"
                cv2.imwrite(filename, imgR)
                filename = temp_path+"imgL"+str(count)+".jpg"
                cv2.imwrite(filename, imgL)                                
            if abs(dx)>threshold: #and this_dx>20:
                print('Using imgR= %s : imgL= %s : dx= %s' % (i,j,dx))
                disp = computeDisparity(imgL, imgR, dx, buff.template, iFlag, debug)
#                if debug:
#                    vis = np.uint8(ip.rescale(disp, (0,255)))
#                    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
#                    cv2.imshow('Disp', vis_color)
#                    cv2.waitKey(500)
                disp = ip.translateImg(disp, (-tdx, 0))  
                sumDisp = sumDisp + disp 
                n += 1
                
#    if debug:
#        cv2.destroyWindow('sumDisp')
    # average disparity
    if n>0:
        avDisp = sumDisp / n
    else:
        avDisp = sumDisp
    
    #rescaling set empirically: cctv = (0.02, 0.25) hdtv = (0, 0.12)
    avDisp = np.uint8(ip.rescale(avDisp, (0,255), (0, 0.1)))
    avDisp[:,-int(dxMax):-1] = 0
    if buff.direction == 'backwards':
        avDisp = np.fliplr(avDisp)
    
    out1 = cv2.applyColorMap(avDisp, cv2.COLORMAP_JET)   
    out2 = ip.overlay(imgRef, avDisp)
    
    
#    glyph = ip.highlight(imgRef, avDisp)
#    cv2.imshow('anaglyph', glyph)
    
#    sumDisp = np.uint8(cctv.rescale(sumDisp, (0,255)))
#    sumDisp[:,-int(dxMax):-1] = 0
#    vis_color = cv2.applyColorMap(sumDisp, cv2.COLORMAP_JET)
#    vis_sum = cctv.imfuse(imgRef, vis_color, 0.2)
#    draw_str(vis_sum, (20, 20), frametxt)
    
    if debug:
        # write results to file
#        filename = temp_path+"Sum"+str(count)+".jpg"
#        cv2.imwrite(filename, vis_sum)
        filename = temp_path+"Mean"+str(count)+".jpg"
        cv2.imwrite(filename, out2)
        
        # display results on screen
#        cv2.imshow('Sum'+str(count), cv2.applyColorMap(sumDisp, cv2.COLORMAP_JET))
#        cv2.imshow('Mean'+str(count), cv2.applyColorMap(avDisp, cv2.COLORMAP_JET))
#        
#        cv2.waitKey(0)
##        cv2.destroyWindow('Sum'+str(count))
#        cv2.destroyWindow('Mean'+str(count))
         
    return out1, out2