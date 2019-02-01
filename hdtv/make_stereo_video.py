#import click
#
#@click.command()
#@click.argument('video_name', type=str)
#@click.option('--lens_only', is_flag=True, default=False)
import os

def process_video(video_name, lens_only, 
                  buffSize = 5, start = 0, stop = 1000, direction = 'backwards', 
                  iFlag = False, debug = False, temp_path = './'):

    import sys
    import cv2
    from dataset import video_hd
    from utils import frame_buffer
    sys.path.append('C:/Users/Mark/opencv-master/samples/python')
    from common import draw_str
    from stereo import stereo_utils
    from utils import image_plotting as ip 
    
    if video_name not in video_hd.VIDEO_NAME_TO_VIDEO:
        print('Could not find video named {}'.format(video_name))
        return

    video = video_hd.VIDEO_NAME_TO_VIDEO[video_name]

    cap = cv2.VideoCapture(video.path)
    if not cap.isOpened():
        print('Could not read video file {}'.format(video.path))
        return

    # Print the frame rate, number of frames and resolution
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print('Frame rate: {}, num frames: {}, shape: {}'.format(fps, num_frames, img_shape))

    # Load lens calibration
    lens_calib = video.lens_calibration
    x, y, w, h = lens_calib.roi

    # Load belt calibration, if provided
    if lens_only:
        belt_calib = None
        # Compute un-distort mapping
        mapping_x, mapping_y = lens_calib.distort_rectify_map(img_shape)
    else:
        belt_calib = video.belt_calibration
        mapping_x, mapping_y = belt_calib.lens_distort_rectilinear_mapping(lens_calib, img_shape)

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
                dst = cv2.remap(img, mapping_x, mapping_y, cv2.INTER_LINEAR)
                if belt_calib is None:
                    # No belt calibration - crop
                    dst = dst[y:y+h, x:x+w]
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
                break
            
            __, __, __ = buff.pop()

        frame_i += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    start = 110
    stop = 120
    debug = True
    temp_path = '../data/belt_images/SUMMER DAWN PD97/'
    # creates a temporary directory to save data generated at runtime
    try:
        os.makedirs(temp_path)
    except OSError:
        if os.path.isdir(temp_path):
            pass
    
    process_video('CQ2014-0GREEN-161125_112316-C3H-004-161130_212953_165', False, 
                  buffSize = 6, 
                  start = start, 
                  stop = stop,
                  direction = 'backwards',
                  iFlag = False,
                  debug = True,
                  temp_path = temp_path)
#    play_video()