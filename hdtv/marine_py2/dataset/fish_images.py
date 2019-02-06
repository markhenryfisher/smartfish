import numpy as np
from pylab import imread, imsave, imshow, show
import os, random, sys, getpass
from glob import glob

from settings import marine_scotland_settings, marine_scotland_videos

from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.color import rgb2grey
from skimage.util import img_as_ubyte

from PIL import Image

import joblib


# Lighting control
LIGHTING_HIGHPASS_SIGMA = 100.0



def _find_file(x):
    filenames = glob(x)
    if len(filenames) == 1:
        return filenames[0]
    elif len(filenames) == 0:
        return None
    else:
        raise RuntimeError('Multiple files {0}'.format(x))


def as_grey(x):
    if len(x.shape) == 3  and  (x.shape[2] == 3 or x.shape[2] == 4):
        return rgb2grey(x)
    else:
        return x



def read_label_png(path):
    img = Image.open(path)
    return np.array(img)

def write_label_png(path, x):
    Image.fromarray(x).save(path)



class FishImage (object):
    def __init__(self, section, name):
        assert isinstance(name, str)
        self.section = section
        self.name = name


    @property
    def video_name(self):
        return self.name.partition('__')[0]

    @property
    def belt_name(self):
        return marine_scotland_videos.VIDEO_NAME_TO_BELT_NAME[self.video_name]


    #
    # Donor path
    #

    @staticmethod
    def input_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'in')

    @staticmethod
    def mask_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'mask')

    @staticmethod
    def edges_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'edges')

    @staticmethod
    def hard_edges_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'hard_edges')

    @staticmethod
    def all_hard_edges_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'hard_edges_all')

    @staticmethod
    def gaussian_edges_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'gaussian_edges')

    @staticmethod
    def multifreqdir_edges_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'edges_dirfreq')

    @staticmethod
    def labels_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'labels')

    @staticmethod
    def all_labels_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'labels_all')

    @staticmethod
    def truth_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'truth')

    @staticmethod
    def multiclass_truth_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'multiclass_truth')

    @staticmethod
    def output_path(section):
        return os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'out')

    @staticmethod
    def output_models_path(section, experiment_type, architecture_name, model_name):
        if architecture_name == '':
            architecture_name = 'default_arch'
        model_dir = os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'output_models', experiment_type, architecture_name)
        model_dir = marine_scotland_settings.ensure_dir_exists(model_dir)
        return os.path.join(model_dir, model_name)

    @staticmethod
    def output_results_path(section, experiment_type, architecture_name):
        if architecture_name == '':
            architecture_name = 'default_arch'
        path = os.path.join(marine_scotland_settings.PROJECT_PATH, section, 'output_results', experiment_type, architecture_name)
        return marine_scotland_settings.ensure_dir_exists(path)


    def has_supervised_data(self):
        return self.mask_image_path() is not None


    #
    # Input images
    #

    def colour_image_path(self):
        return _find_file(os.path.join(self.input_path(self.section), '{0}.png'.format(self.name)))

    def read_colour_image(self):
        col_2d = imread(self.colour_image_path())
        if col_2d.shape[2] == 4:
            col_2d = col_2d[:,:,:3]
        return col_2d.astype(np.float32)

    def get_colour_image_shape(self):
        img = Image.open(self.colour_image_path())
        width, height = img.size
        return height, width, 3


    def mask_image_path(self):
        return _find_file(os.path.join(self.mask_path(self.section), '{0}.png'.format(self.name)))

    def read_mask_image(self):
        return as_grey(imread(self.mask_image_path())).astype(np.float32)


    def edges_image_path(self):
        return _find_file(os.path.join(self.edges_path(self.section), '{0}.npy'.format(self.name)))

    def read_edges_image(self):
        return np.load(self.edges_image_path()).astype(np.float32)


    def hard_edges_image_path(self):
        return _find_file(os.path.join(self.hard_edges_path(self.section), '{0}.png'.format(self.name)))

    def read_hard_edges_image(self):
        return as_grey(imread(self.hard_edges_image_path())).astype(np.float32)


    def all_hard_edges_image_path(self):
        return _find_file(os.path.join(self.all_hard_edges_path(self.section), '{0}.png'.format(self.name)))

    def read_all_hard_edges_image(self):
        return as_grey(imread(self.all_hard_edges_image_path())).astype(np.float32)


    def gaussian_edges_image_path(self):
        return _find_file(os.path.join(self.gaussian_edges_path(self.section), '{0}.npy'.format(self.name)))

    def read_gaussian_edges_image(self):
        return np.load(self.gaussian_edges_image_path())


    def multifreqdir_edges_image_path(self):
        return _find_file(os.path.join(self.multifreqdir_edges_path(self.section), '{0}.npz'.format(self.name)))

    def read_multifreqdir_edges_image(self):
        return np.load(self.multifreqdir_edges_image_path())['arr_0']


    def label_image_path(self):
        return _find_file(os.path.join(self.labels_path(self.section), '{0}.png'.format(self.name)))

    def read_label_image(self):
        return read_label_png(self.label_image_path())

    def write_label_image(self, x):
        write_label_png(self.label_image_path(), x)


    def all_labels_image_path(self):
        return _find_file(os.path.join(self.all_labels_path(self.section), '{0}.png'.format(self.name)))

    def read_all_labels_image(self):
        return read_label_png(self.all_labels_image_path())

    def write_all_labels_image(self, x):
        write_label_png(self.all_labels_image_path(), x)


    def truth_image_path(self):
        return _find_file(os.path.join(self.truth_path(self.section), '{0}.png'.format(self.name)))

    def read_truth_image(self):
        return as_grey(imread(self.truth_image_path())).astype(np.float32)


    def multiclass_truth_image_path(self):
        return _find_file(os.path.join(self.multiclass_truth_path(self.section), '{0}.png'.format(self.name)))

    def read_multiclass_truth_image(self):
        return read_label_png(self.multiclass_truth_image_path())


    def output_image_path(self):
        return os.path.join(self.output_path(self.section), '{0}.png'.format(self.name))


    def models_path(self, experiment_type, architecture_name, model_name):
        return self.output_models_path(self.section, experiment_type, architecture_name, model_name)

    def results_path(self, experiment_type, architecture_name, suffix, ext=''):
        return os.path.join(self.output_results_path(self.section, experiment_type, architecture_name), '{}_{}{}'.format(self.name, suffix, ext))

    def read_results(self, experiment_type, architecture_name, suffix):
        return joblib.load(self.results_path(experiment_type, architecture_name, suffix))

    def write_results(self, experiment_type, architecture_name, suffix, data):
        joblib.dump(data, self.results_path(experiment_type, architecture_name, suffix))



    def __str__(self):
        return '({0}::{1})'.format(self.section, self.name)

    def __repr__(self):
        return 'FishImage({0}, {1})'.format(repr(self.section), repr(self.name))





ALL_INPUT_IDS = [('segmentation_test_data', os.path.splitext(os.path.basename(x))[0])   for x in os.listdir(FishImage.input_path('segmentation_test_data'))   if not x.startswith('.')]
ALL_IMAGES = [FishImage(*x)   for x in ALL_INPUT_IDS]

ALL_WARPED_INPUT_IDS = [('warpseg_test_data', os.path.splitext(os.path.basename(x))[0])   for x in os.listdir(FishImage.input_path('warpseg_test_data'))   if not x.startswith('.')]
ALL_WARPED_IMAGES = [FishImage(*x)   for x in ALL_WARPED_INPUT_IDS]

ALL_BOAT_WARPED_INPUT_IDS = [('allboat_warpseg_test_data', os.path.splitext(os.path.basename(x))[0])   for x in os.listdir(FishImage.input_path('allboat_warpseg_test_data'))   if not x.startswith('.')]
ALL_BOAT_WARPED_IMAGES_ALL = [FishImage(*x)   for x in ALL_BOAT_WARPED_INPUT_IDS]
ALL_BOAT_WARPED_IMAGES = [fish_img for fish_img in ALL_BOAT_WARPED_IMAGES_ALL if fish_img.has_supervised_data()]
ALL_BOAT_WARPED_IMAGES_UNSUP = [fish_img for fish_img in ALL_BOAT_WARPED_IMAGES_ALL if not fish_img.has_supervised_data()]




def fit_standardiser(standardiser, images):
    if standardiser.has_been_fit:
        print('Not fitting standardiser; already fit')
    else:
        standardiser.fit([img.read_colour_image() for img in images], [img.belt_name for img in images])