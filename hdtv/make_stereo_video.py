#import click
#
#@click.command()
#@click.argument('video_name', type=str)
#@click.option('--lens_only', is_flag=True, default=False)


def process_video(video_name, lens_only, *args):
    import sys
    import cv2
    from dataset import video_hd
    from utils import frame_buffer
    sys.path.append('C:/Users/Mark/opencv-master/samples/python')
    from common import draw_str
    
    buffSize = args[0]
    start = args[1]
    stop = args[2]

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

    buff = frame_buffer.FrameBuffer(buffSize)
    
    for i in range(buffSize):
        success, img = cap.read()
        if not success:
            raise ValueError('Failed to read video frame')
        dst = cv2.remap(img, mapping_x, mapping_y, cv2.INTER_LINEAR)
        if belt_calib is None:
            # No belt calibration - crop
            dst = dst[y:y+h, x:x+w]
        buff.push(dst)

    frame_i = 0
    while True:      
        f = buff.data[-1]
        x = buff.x[-1]
        
        vis = f.copy()
        draw_str(vis, (20, 20), 'Frame: %d D: %.2f' %(frame_i,x))

        cv2.imshow('Un-distorted', vis)

        sys.stdout.write('\rFrame {}'.format(frame_i))

        if buff.direction is None or frame_i < start:
            k = cv2.waitKey(1)
        else:
            k = cv2.waitKey(1000)             
            
        if k == 27 or frame_i >= stop:
            break

        __, __ = buff.pop()
        success, img = cap.read()
        if not success:
            raise ValueError('Failed to read video frame')
        dst = cv2.remap(img, mapping_x, mapping_y, cv2.INTER_LINEAR)
        if belt_calib is None:
            # No belt calibration - crop
            dst = dst[y:y+h, x:x+w]
        buff.push(dst)
        frame_i += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video('CQ2014-0GREEN-161125_112316-C3H-004-161130_212953_165', False, 5, 10, 20)
#    play_video()