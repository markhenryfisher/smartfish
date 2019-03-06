#import click
#
#@click.command()
#@click.argument('video_name', type=str)
#@click.option('--lens_only', is_flag=True, default=False)
def play_video(video_name, lens_only):
    import sys
    import cv2
    from dataset import video_hd

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


    frame_i = 0
    while True:
        success, img = cap.read()

        if not success:
            break

        dst = cv2.remap(img, mapping_x, mapping_y, cv2.INTER_LINEAR)

        if belt_calib is None:
            # No belt calibration - crop
            dst = dst[y:y+h, x:x+w]

        cv2.imshow('Un-distorted', dst)

        sys.stdout.write('\rFrame {}'.format(frame_i))

        k = cv2.waitKey(0) #1
        if k == 27:
            break

        frame_i += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
  # MRV Scotia Example
  play_video('vlc-record-2018-05-30-14h32m23s-ABSENT-ABSENT-180122_141435-C4H-141-180204_085409_188.MP4-', False)
  # Summer Dawn Example
#  play_video('CQ2014-0GREEN-161125_112316-C3H-004-161130_212953_165', False)
#    play_video()