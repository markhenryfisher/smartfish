# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:54:30 2019

@author: Mark
"""

import os

_DATA_PATHS = [
    r'C:\fish\data\fish2014mhf', # mhf fix
]

_SCRATCH_PATHS = [
    r'C:\scratch\data\fish2014mhf', # mhf fix
]

DATA_PATH = None
SCRATCH_PATH = None

for p in _DATA_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise RuntimeError('Could not locate project data path; tried {}'.format(_DATA_PATHS))

for p in _SCRATCH_PATHS:
    if os.path.exists(p):
        SCRATCH_PATH = p
        break

if SCRATCH_PATH is None:
    raise RuntimeError('Could not locate project scratch path; tried {}'.format(_SCRATCH_PATHS))

def ensure_dir_exists(path):
    """
    Ensure that the specified directory exists; create it if it does not

    :param path: the path to check

    :return: `path`
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

CALIBRATION_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'calibration'))
VIDEO_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'CCTV-VIDEO'))
RESULTS_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'results'))
DEBUG_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'debug'))