#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19.03.19 - stereo_utils now recovers 3d model (xyz) using camera_matrix. Will need to add fix for Harvester at some point in the future.  
11.03.19 - Changed get belt motion.
08.03.19 - rechecking baseline (doesn't make much difference)
07.03.19 - testing MRV SCOTIA. Mod. to set sgbm parameters depending on belt.
06.03.19 - branch @revision 31
04.03.19 - now writing all results and debug to C:\fish\data... video output filename is
            auto generated using datetime. 
01.03.19 - added support for both json and yml calibration files. If calibration file is specified then 
            we use MHF's recification code, otherwise Geoff's.
11.02.19 - added 'reset' feature to frame buffer.
09.02.19 - replaced getBeltTravelByOpticalFlow() with getBeltTravelByTemplateMatching()
            Note: Also adopted convention that belt distance is +ve for left to right travel
31.01.19 - results displayed as 'montage'
30.01.19 - completely rewritten in same style as hdtv make_stereo_video
11.01.19 - introduced this_dx to fix problems with frames 73, 74 etc.
10.01.19 - simplified how dispSum is computed. Changed how disparity is normalised.
            increased buffer size to 7 (gives ~ 150mm belt motion).
            Added function to check belt motion for consistency.
03.01.19 - removed template from preprocessing step. Changed way disparity is rendered. 
02.01.19 - changes to computeDisparity to give 'normalised' disparity,
            and messed with the colormaps
22.12.18 - changes to processDisparity to calculate many disparity estimates.
20.12.18 - copied to cctv_processVideo
Script to test cctv video input, FrameBuffer, Disparity
@filename: processVideo_v1.py
@author: mark.fisher@uea.ac.uk
@last_updated: 20.12.18
"""

import os
global minViableStereoBaseline
minViableStereoBaseline_vga = 70
minViableStereoBaseline_hd = 30
      
def process_video(video_name, 
                  buffSize = 5, start = 0, stop = 1000, direction = 'forwards', 
                  iFlag = False, debug = False):
    
    from dataset import video_mhf
    from utils import frame_buffer
    from utils import cal_utils
    from stereo import stereo_utils
    from utils import image_plotting as ip 
    import sys
    sys.path.append('C:/Users/Mark/opencv-master/samples/python')
    from common import draw_str
    import cv2
    import numpy as np
    import datetime

    if video_name not in video_mhf.VIDEO_NAME_TO_VIDEO:
        print('Could not find video named {}'.format(video_name))
        return
    
    video = video_mhf.VIDEO_NAME_TO_VIDEO[video_name]
    
    cap = cv2.VideoCapture(video.path)
    if not cap.isOpened():
        print('Could not read video file {}'.format(video.path))
        return
    
    # Print the frame rate, number of frames and resolution
    video.fps = cap.get(cv2.CAP_PROP_FPS)
    video.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video.img_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print('Frame rate: {}, num frames: {}, shape: {}'.format(video.fps, video.num_frames, video.img_shape))

    

    # Load lens calibration (in this case lens & beltROI)
    lens_calib = video.lens_calibration
    if type(lens_calib).__name__ == 'Bunch': # its mhf's 
        pass
    else:
        x, y, w, h = lens_calib.roi # it's Geoff's
        belt_calib = video.belt_calibration
        mapping_x, mapping_y = belt_calib.lens_distort_rectilinear_mapping(lens_calib, video.img_shape)


    buff = frame_buffer.FrameBuffer(buffSize, direction, video)
    outvidfilename = None
    
    if start > 0:
        print('\nSpooling to Frame {}...'.format(start))

    out1 = out2 = cv2.applyColorMap(np.zeros(video.img_shape, dtype=np.ubyte), cv2.COLORMAP_JET)
    frame_i = 0
    
    while True:
        if not debug:
            sys.stdout.write('\rFrame {}'.format(frame_i))
        else:
            print('Frame {}'.format(frame_i))
                

        if frame_i < start:
            success, img = cap.read()
            if not success:
                raise ValueError('Failed to read video frame')         
        else:
            while buff.count < buff.size:
                success, img = cap.read()
                if not success:
                    raise RuntimeError('Failed to read video frame')
                if type(lens_calib).__name__ == 'Bunch': # its mhf's 
                    dst = cal_utils.rectify(img, lens_calib)
                else:    # Geoff's
                    dst = cv2.remap(img, mapping_x, mapping_y, cv2.INTER_LINEAR)
                    if belt_calib is None:
                        # No belt calibration - crop
                        dst = dst[y:y+h, x:x+w]
                        
                buff.push(img, dst)
            
            r = buff.raw[-1]
            f = buff.data[-1]
            x = buff.x[-1]
            
            # compute disparity if sufficient stereo baseline
            if buff.sufficientStereoBaseline:
                out1, out2 = stereo_utils.process_frame_buffer(buff, frame_i, iFlag, debug, video.belt.debug_dir)
            else:
                print('Insufficient Stereo Baseline!')

#else:
#                out1 = ip.translateImg(out1, (buff.getLastdx(), 0))
#                out2 = ip.translateImg(out2, (buff.getLastdx(), 0))
#            
            # gather image frames and montage
            vis0 = r.copy()
            draw_str(vis0, (20, 20), 'Frame: %d D: %.2f' %(frame_i,x))
            vis1 = f.copy()
            draw_str(vis1, (20, 20), 'Rectified')
            vis2 = out1.copy()
            draw_str(vis2, (20,20), 'Depth')
            vis3 = out2.copy()
            frametxt = "Buff: %s-%s; Largest Stereo Baseline: %s." % (frame_i,frame_i+buff.nItems()-1,round(buff.getLargestStereoBaseline()))    
            draw_str(vis3, (20, 20), frametxt)

            out_frame = ip.montage(2,2,(640, 480), vis0, vis1, vis2, vis3)

            if outvidfilename is None:
                frame_height, frame_width = out_frame.shape[:2]
                outvidfilename = os.path.join(video.belt.results_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.avi')
                out = cv2.VideoWriter(outvidfilename,cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))

            cv2.imshow('Stereo', out_frame)
            out.write(out_frame)
            
            k = cv2.waitKey(0)          
            if k == 27 or frame_i >= stop:
                cap.release()
                out.release()
                break
            
            __, __, __ = buff.pop()

        frame_i += 1

    cv2.destroyAllWindows()
    
    
def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='process video to find stereo disparity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_name', type=str, default="Belt E base",
                        help='video name.')
    parser.add_argument('--start', type=int, default=0, help='Start at frame=start_idx')
    parser.add_argument('--stop', type=int, default=500, help='Stop at frame=stop_idx')
    args = parser.parse_args()
    
    return args
        

if __name__ == '__main__':

    args = parse_args()
    
    video_name = args.video_name
    video_name = 'vlc-record-2018-05-30-14h32m23s-ABSENT-ABSENT-180122_141435-C4H-141-180204_085409_188.MP4-'
    
    # Note: Ensure buffSize is large enough to give sufficient stereo baseline and direction of belt is set correctly (forwards is left-to-right).
    process_video(video_name, 
              buffSize = 6, 
              start = args.start, 
              stop = args.stop,
              direction = 'backwards',
              iFlag = False,
              debug = True)
    
    
    
            