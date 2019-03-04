# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:39:41 2019

@author: Mark
"""
import os
import json

from settings import mhf_settings 

class Belt (object):
    def __init__(self, belt_name):
        self.name = belt_name

        # Video
        self.videos_dir = mhf_settings.ensure_dir_exists(
            os.path.join(mhf_settings.VIDEO_DIR, self.name))
        
        # Calibration
        self.calib_dir = mhf_settings.ensure_dir_exists(
            os.path.join(mhf_settings.CALIBRATION_DIR, self.name))
        self.__calib_cache_lens = {}
        self.__calib_cache_belt = {}
        self.__calib_cache_belt_realign = {}
        
    def lens_calibration_path(self, camera_name):
        return os.path.join(self.calib_dir, 'lens__camera_{}'.format(camera_name))
    
    def get_lens_calibration(self, camera_name):
        key = camera_name
        try:
            return self.__calib_cache_lens[key]
        except KeyError:
            path = self.lens_calibration_path(camera_name)
            if os.path.exists(path+'.yml'):
                from utils import cal_utils # temp fix
                x = cal_utils.getCameraParams(path+'.yml') # temp fix
            elif os.path.exists(path+'.json'): 
                from calibration import lens
                with open(path, 'r') as f:
                    calib_js = json.load(f)
                x = lens.LensCalibration.from_json(calib_js)
            else:
                raise RuntimeError('Could not find lens calibration file {} for camera {}'.format(path, camera_name))    
            self.__calib_cache_lens[key] = x
            return x

        
BELTS = []
BELT_NAME_TO_BELT = {}
filename = None
belt = None
for filename in os.listdir(mhf_settings.VIDEO_DIR):
    if os.path.isdir(os.path.join(mhf_settings.VIDEO_DIR, filename)):
        belt = Belt(filename)
        BELTS.append(belt)
        BELT_NAME_TO_BELT[filename] = belt
del filename, belt

class VideoLITE (object):
    """VideoLITE - A lightweight class for organising mhf cctv video"""

    def __init__(self, name, filename, path, belt):
        self.name = name
        self.filename = filename
        self.path = path 
        self.belt = belt
        self.camera_name = "default" #None
        
        self.__lens_calib = None
        
    @property
    def lens_calibration(self):
        if self.__lens_calib is None:
            if self.camera_name is None:
                raise RuntimeError('Cannot get lens calibration for video {} as no camera name is registered; run the `register_calibration` program'.format(self.name))
            self.__lens_calib = self.belt.get_lens_calibration(self.camera_name)
        return self.__lens_calib
        
        
VIDEOS = []
VIDEO_NAME_TO_VIDEO = {}
filename = name = None
belt = video = None
for belt in BELTS:
    for filename in os.listdir(belt.videos_dir):
        if os.path.isfile(os.path.join(belt.videos_dir, filename)):
            name = os.path.splitext(filename)[0]
            path = os.path.join(belt.videos_dir, filename)
            # Ignore calibration videos
            if not name.startswith('calibration') and not name.startswith('Calibration') and not name.endswith('__motion'):
                video = VideoLITE(name, filename, path, belt)
                VIDEOS.append(video)
                VIDEO_NAME_TO_VIDEO[name] = video
del filename, name, video