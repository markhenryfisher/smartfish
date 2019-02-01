import math

import numpy as np
import scipy.io
from skimage import transform
try:
    import cv2
except ImportError:
    cv2 = None

from settings import marine_scotland_videos


DEFAULT_PIXELS_PER_METRE = 450.0


class BeltTransform (object):
    """
    Belt parameters

    Attributes:-

    corners: a 4x2 array of co-ordinates of the corners, ordered:
        belt start bottom, belt start top, belt end top, beld end bottom
    physical_size_m: a array of [width, length] specifying the physical dimensions of the belt in metres
    physical_aspect_ratio: physical length / physical width
    pixels_per_metre: the number of pixels per metre (provided as a parameter to the constructor)
    physical_size_px - a tuple of (width, length) as integers, of the size of the belt in pixels;
        product of physical_size_m and pixels_per_metre
    """


    def __init__(self, corners, physical_size_m, pixels_per_metre=DEFAULT_PIXELS_PER_METRE):
        self.corners = corners
        # The physical sizes were guessed by eye.
        # There are more accurate estimates for the width in marine_scotland_videos
        self.physical_size_m = physical_size_m
        self.physical_aspect_ratio = self.physical_size_m[1] / self.physical_size_m[0]
        self.pixels_per_metre = pixels_per_metre
        self.physical_size_px =  np.ceil(np.array([self.physical_size_m[0] * self.pixels_per_metre,
                                                   self.physical_size_m[1] * self.pixels_per_metre])).astype(int)

        physical_flat_coord = np.array([[0, self.physical_size_px[0]],
                                        [0, 0],
                                        [self.physical_size_px[1], 0],
                                        [self.physical_size_px[1], self.physical_size_px[0]]])

        self.physical_proj = transform.ProjectiveTransform()
        self.physical_proj.estimate(self.corners, physical_flat_coord)


    @staticmethod
    def from_matlab_file_data(belt_name, mat, pixels_per_metre=DEFAULT_PIXELS_PER_METRE):
        belt = mat['belt_parameters']

        corners = belt['screen_coord'][0][0]
        # The physical sizes were guessed by eye.
        # There are more accurate estimates for the width in marine_scotland_videos
        est_width = marine_scotland_videos.BELT_WIDTH_ESTIMATES[belt_name]
        physical_size_by_eye = np.array([belt['width'][0][0][0][0], belt['length'][0][0][0][0]])
        aspect_ratio_by_eye = physical_size_by_eye[1] / physical_size_by_eye[0]
        physical_size_m = np.array([est_width, est_width * aspect_ratio_by_eye])

        return BeltTransform(corners, physical_size_m, pixels_per_metre=pixels_per_metre)


    @staticmethod
    def from_json(belt_json, pixels_per_metre=DEFAULT_PIXELS_PER_METRE):
        corners = np.array([
            [belt_json['exit_bottom']['x'], belt_json['exit_bottom']['y']],
            [belt_json['exit_top']['x'], belt_json['exit_top']['y']],
            [belt_json['entrance_top']['x'], belt_json['entrance_top']['y']],
            [belt_json['entrance_bottom']['x'], belt_json['entrance_bottom']['y']]
            ])

        physical_size_m = np.array([
            belt_json['width'], belt_json['length']
        ])

        return BeltTransform(corners, physical_size_m, pixels_per_metre=pixels_per_metre)



    def physical_transform_and_size_px(self, pixels_per_metre=None):
        if pixels_per_metre is None  or  pixels_per_metre == self.pixels_per_metre:
            return self.physical_proj, self.physical_size_px
        else:
            physical_size_px =  np.ceil(np.array([self.physical_size_m[0] * pixels_per_metre,
                                                  self.physical_size_m[1] * pixels_per_metre])).astype(int)

            physical_flat_coord = np.array([[0, physical_size_px[0]],
                                            [0, 0],
                                            [physical_size_px[1], 0],
                                            [physical_size_px[1], physical_size_px[0]]])

            proj = transform.ProjectiveTransform()
            proj.estimate(self.corners, physical_flat_coord)

            return proj, physical_size_px




    def physical_warp(self, img, pixels_per_metre=None, order=1):
        proj, sz_px = self.physical_transform_and_size_px(pixels_per_metre=pixels_per_metre)
        if cv2 is not None:
            warped = cv2.warpPerspective(img, proj.params, (sz_px[1], sz_px[0]))
        else:
            warped = transform.warp(img, proj.inverse, order=order)[:int(sz_px[0]),:int(sz_px[1])]
        return warped.astype('float32')

    def inv_physical_warp(self, img, pixels_per_metre=None, order=1):
        proj, sz_px = self.physical_transform_and_size_px(pixels_per_metre=pixels_per_metre)
        if cv2 is not None:
            warped = cv2.warpPerspective(img, np.linalg.inv(proj.params), (img.shape[1], img.shape[0]))
        else:
            warped = transform.warp(img, proj, order=order)
        return warped.astype('float32')



    def warped_labelled_image(self, limg):
        proj, sz_px = self.physical_transform_and_size_px()
        return limg.warped(proj, sz_px)


    @staticmethod
    def from_path(belt_name, mat_path):
        mat = scipy.io.loadmat(mat_path)
        return BeltTransform.from_matlab_file_data(belt_name, mat)


    @staticmethod
    def for_video(video_name):
        path = marine_scotland_videos.belt_params_path(video_name)
        belt_name = marine_scotland_videos.VIDEO_NAME_TO_BELT_NAME[video_name]
        return BeltTransform.from_path(belt_name, path)


    @staticmethod
    def for_belt(belt_name):
        path = marine_scotland_videos.belt_params_path(belt_name=belt_name)
        return BeltTransform.from_path(belt_name, path)


