#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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
        
def process_video(video_filename, cal_filename, 
                  buffSize = 5, start = 0, stop = 1000, direction = 'backwards', 
                  iFlag = False, debug = False, temp_path = './'):
    
    from utils import frame_buffer
    from utils import cal_utils
    from stereo import stereo_utils
    from utils import image_plotting as ip 
    import sys
    sys.path.append('C:/Users/Mark/opencv-master/samples/python')
    from common import draw_str
    import cv2
    
    cameraParams = cal_utils.getCameraParams(cal_filename)
  
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print('Could not read video file {}'.format(video_filename))
        return
    
    # Print the frame rate, number of frames and resolution
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print('Frame rate: {}, num frames: {}, shape: {}'.format(fps, num_frames, img_shape))

    buff = frame_buffer.FrameBuffer(buffSize, direction, temp_path)
    outvidfilename = None
    if start > 0:
        print('\nSpooling to Frame {}...'.format(start))
    frame_i = 0
    while True:
        sys.stdout.write('\rFrame {}'.format(frame_i))

        if frame_i < start:
            success, img = cap.read()
            if not success:
                raise ValueError('Failed to read video frame')         
        else:
            while buff.count < buff.size:
                success, img = cap.read()
                if not success:
                    raise ValueError('Failed to read video frame')
#                dst = cv2.remap(img, mapping_x, mapping_y, cv2.INTER_LINEAR)
#                if belt_calib is None:
#                    # No belt calibration - crop
#                    dst = dst[y:y+h, x:x+w]
                dst = cal_utils.rectify(img, cameraParams)
                buff.push(img, dst)
            
            r = buff.raw[-1]
            f = buff.data[-1]
            x = buff.x[-1]
            out1, out2 = stereo_utils.process_frame_buffer(buff, frame_i, iFlag, debug, temp_path)
            
            # gather image frames and montage
            vis0 = r.copy()
            draw_str(vis0, (20, 20), 'Frame: %d D: %.2f' %(frame_i,x))
            vis1 = f.copy()
            draw_str(vis1, (20, 20), 'Rectified')
            vis2 = out1.copy()
            vis3 = out2.copy()
            out_frame = ip.montage(2,2,(640, 480), vis0, vis1, vis2, vis3)

            if outvidfilename is None:
                frame_height, frame_width = out_frame.shape[:2]
                outvidfilename = temp_path+'outpy.avi'
                out = cv2.VideoWriter(outvidfilename,cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))

            cv2.imshow('Stereo', out_frame)
            out.write(out_frame)
            
            k = cv2.waitKey(1000)             
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
    parser.add_argument('--root_path', type=str, default="../data/",
                        help='Root pathname.')
    parser.add_argument('--video_file', type=str, default="Belt E base.mp4",
                        help='video filename.')
    parser.add_argument('--template_file', type=str, default="template.tif",
                        help='template filename.')
    parser.add_argument('--cal_file', type=str, default="beltE/cameraParams.yml",
                        help='calibration filename.')
    parser.add_argument('--start', type=int, default=55, help='Start at frame=start_idx')
    parser.add_argument('--stop', type=int, default=300, help='Stop at frame=stop_idx')
    args = parser.parse_args()
    
    return args
        

if __name__ == '__main__':

    args = parse_args()

    
    # creates a temporary directory to save data generated at runtime
    temp_path = args.root_path+'temp/'
    try:
        os.makedirs(temp_path)
    except OSError:
        if os.path.isdir(temp_path):
            pass
    
    
    process_video(args.root_path+args.video_file, args.root_path+args.cal_file, 
              buffSize = 7, 
              start = args.start, 
              stop = args.stop,
              direction = 'forwards',
              iFlag = False,
              debug = True,
              temp_path = temp_path)
    
    
    
            