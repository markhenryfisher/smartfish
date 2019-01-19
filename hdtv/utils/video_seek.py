import cv2



def video_seek_start(cap, buggy=True):
    """
    Video seeking helper

    Moves the current position of an OpenCV VideoCapture to the start

    Uses either:

    `cap.set(cv2.CAP_PROP_POS_MSEC, 0)`: works properly

    Or;

    `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)`: buggy

    :param cap: VideoCapture object
    :param buggy: Use `cv2.CAP_PROP_POS_FRAMES` if True, `cv2.CAP_PROP_POS_MSEC` otherwise
    :return: `cap`
    """
    if buggy:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    return cap

