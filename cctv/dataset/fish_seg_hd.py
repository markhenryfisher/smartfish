import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from skimage.util import img_as_float
from skimage.color import rgb2grey
import cv2

from settings import ms2016_settings
from dataset import training_dataset, video_hd


def as_grey(x):
    if len(x.shape) == 3  and  (x.shape[2] == 3 or x.shape[2] == 4):
        return rgb2grey(x)
    else:
        return x

_IMAGE_NAME_PATTERN = re.compile(r'(.+)__(\d+)__(\d+)')

def image_name_to_metadata(image_name):
    match = _IMAGE_NAME_PATTERN.match(image_name)
    if match is not None:
        video_name = match.group(1)
        frame_index = int(match.group(2))
        frame_choice_index = int(match.group(3))
        return video_name, frame_index, frame_choice_index
    else:
        return None, None, None


#
# IMAGE ACCESSOR
#

class FishImage (object):
    def __init__(self, belt, image_name):
        self.belt = belt
        self.name = image_name
        self.__image_shape = None

        match = _IMAGE_NAME_PATTERN.match(image_name)
        if match is not None:
            self.video_name = match.group(1)
            self.frame_index = int(match.group(2))
            self.frame_choice_index = int(match.group(3))

            self.video = video_hd.VIDEO_NAME_TO_VIDEO.get(self.video_name)
        else:
            self.video_name = None
            self.frame_index = None
            self.frame_choice_index = None
            self.video = None


    def __eq__(self, other):
        if isinstance(other, FishImage):
            return self.name == other.name and self.belt == other.belt
        else:
            return False

    def __hash__(self):
        return hash((FishImage, self.belt, self.name))

    def __repr__(self):
        return 'FishImage({}: {})'.format(self.belt, self.name)

    def __str__(self):
        return 'FishImage({}: {})'.format(self.belt, self.name)


    @property
    def cam_image_path(self):
        return os.path.join(self.belt.seg_label_frames_dir, '{}.png'.format(self.name))

    @property
    def input_image_path(self):
        return os.path.join(self.belt.seg_input_image_dir, '{}.png'.format(self.name))

    @property
    def seg_labels_json_path(self):
        return os.path.join(self.belt.seg_labels_json_dir, '{}__labels.json'.format(self.name))

    @property
    def truth_fg_image_path(self):
        return os.path.join(self.belt.seg_truth_fg_dir, '{}.png'.format(self.name))

    @property
    def truth_mask_image_path(self):
        return os.path.join(self.belt.seg_truth_mask_dir, '{}.png'.format(self.name))

    @property
    def truth_edges_image_path(self):
        return os.path.join(self.belt.seg_truth_edges_dir, '{}.png'.format(self.name))

    @property
    def truth_labels_image_path(self):
        return os.path.join(self.belt.seg_truth_labels_dir, '{}.png'.format(self.name))

    @property
    def truth_deep_watershed_path(self):
        return os.path.join(self.belt.seg_deep_watershed_dir, '{}.npz'.format(self.name))

    @property
    def truth_maximal_ball_normal_map_path(self):
        return os.path.join(self.belt.seg_maximal_ball_normal_map_dir, '{}.npz'.format(self.name))


    @property
    def species_id_labels_json_path(self):
        return os.path.join(self.belt.species_id_labels_json_dir, '{}__labels.json'.format(self.name))

    @property
    def species_id_ground_truth_path(self):
        return os.path.join(self.belt.species_id_ground_truth_dir, '{}.npz'.format(self.name))



    def predictions_path(self, dir_names, filename_suffix_and_ext):
        filename = '{}__{}'.format(self.name, filename_suffix_and_ext)
        names = list(dir_names) + [self.belt.name, filename]
        return os.path.join(ms2016_settings.PREDICTIONS_DIR, *names)

    def output_predictions_path(self, dir_names, filename_suffix_and_ext):
        filename = '{}__{}'.format(self.name, filename_suffix_and_ext)
        dir_names = list(dir_names) + [self.belt.name]
        pred_dir = ms2016_settings.ensure_dir_exists(os.path.join(ms2016_settings.PREDICTIONS_DIR, *dir_names))
        return os.path.join(pred_dir, filename)


    def has_ground_truth(self):
        return os.path.exists(self.truth_fg_image_path)


    @property
    def image_size(self):
        img = Image.open(self.input_image_path)
        width, height = img.size
        return height, width

    def read_cam_image(self):
        img = cv2.imread(self.cam_image_path)
        return img_as_float(img[:, :, ::-1])

    @property
    def cam_image_size(self):
        img = Image.open(self.cam_image_path)
        width, height = img.size
        return height, width

    def read_input_image(self):
        img = cv2.imread(self.input_image_path)
        return img_as_float(img[:, :, ::-1])

    def read_truth_fg_image(self):
        img = cv2.imread(self.truth_fg_image_path)
        return img_as_float(as_grey(img))

    def read_truth_mask_image(self):
        img = cv2.imread(self.truth_mask_image_path)
        return img_as_float(as_grey(img))

    def read_truth_edges_image(self):
        img = cv2.imread(self.truth_edges_image_path)
        return img_as_float(as_grey(img))

    def read_truth_labels_image(self):
        img = Image.open(self.truth_labels_image_path)
        return np.array(img)

    def read_deep_watersheed_dist_and_dir_map(self):
        f = np.load(self.truth_deep_watershed_path)
        dist_map = f['dist_map']
        dir_map = f['dir_map']
        return dist_map, dir_map

    def read_maximal_ball_normal_map(self):
        f = np.load(self.truth_maximal_ball_normal_map_path)
        normal_map = f['normal_map']
        return normal_map





_ALL_IMAGES = None
_SUPERVISED_IMAGES = None
_UNSUPERVISED_IMAGES = None


def _refresh_images():
    global _ALL_IMAGES, _SUPERVISED_IMAGES, _UNSUPERVISED_IMAGES
    if _ALL_IMAGES is None:
        _ALL_IMAGES = []
        _SUPERVISED_IMAGES = []
        _UNSUPERVISED_IMAGES = []
        for belt in video_hd.BELTS:
            for filename in os.listdir(belt.seg_input_image_dir):
                name, ext = os.path.splitext(filename)
                if ext.lower() == '.png':
                    img = FishImage(belt, name)
                    _ALL_IMAGES.append(img)
                    if img.has_ground_truth():
                        _SUPERVISED_IMAGES.append(img)
                    else:
                        _UNSUPERVISED_IMAGES.append(img)


def get_all_images():
    _refresh_images()
    return _ALL_IMAGES

def get_supervised_images():
    _refresh_images()
    return _SUPERVISED_IMAGES

def get_unsupervised_images():
    _refresh_images()
    return _UNSUPERVISED_IMAGES



#
# DATASET
#


_FOLD_PATTERN = re.compile('Fold (\d+)')


class FishDataset(object):
    def __init__(self, belt_to_xv, belt_to_samples):
        n_folds = None
        for xv in belt_to_xv.values():
            n_folds = xv.n_folds
            break

        self.belt_to_xv = belt_to_xv
        self.belt_to_samples = belt_to_samples
        self.belts = list(sorted(belt_to_xv.keys(), key=lambda belt: belt.name))
        self.n_folds = n_folds

        samples = []
        for belt in self.belts:
            samples.extend(self.belt_to_samples[belt])
        self.all_samples = samples



    def get_belt(self, belt):
        return self.belt_to_xv[belt]

    def get_fold(self, belt, fold_i):
        return self.belt_to_xv[belt].datasets[fold_i]

    def get_samples_in_belt(self, belt):
        return self.belt_to_samples[belt]


    @staticmethod
    def create(belts=None, samples=None, n_folds=5, rng=None):
        if belts is None:
            belts = video_hd.BELTS

        if samples is None:
            samples = get_supervised_images()

        belt_to_xv = {}
        belt_to_samples = {}

        for belt in belts:
            samples_in_belt = [s for s in samples if s.belt == belt]
            xv = training_dataset.CrossValidation.from_samples(samples_in_belt, n_folds=n_folds, rng=rng)
            belt_to_xv[belt] = xv
            belt_to_samples[belt] = samples_in_belt

        return FishDataset(belt_to_xv, belt_to_samples)


    def to_df(self):
        all_image_names = [img.name for img in self.all_samples]

        # Build dataframe, with column for belt name and a column for each fold
        columns = ['Belt'] + ['Fold {}'.format(fold_i) for fold_i in range(self.n_folds)]
        df = pd.DataFrame(index=all_image_names, columns=columns)

        # Fill in belt column for each row
        for img in self.all_samples:
            df.loc[img.name, 'Belt'] = img.belt.name

        # Fill in splits
        for belt in self.belts:
            xv = self.belt_to_xv[belt]
            for fold_i, ds in enumerate(xv.datasets):
                for img in ds.train:
                    df.loc[img.name,'Fold {}'.format(fold_i)] = 'Train'
                for img in ds.validation:
                    df.loc[img.name,'Fold {}'.format(fold_i)] = 'Val'
                for img in ds.test:
                    df.loc[img.name,'Fold {}'.format(fold_i)] = 'Test'

        return df


    def to_h5(self, h5_path):
        df = self.to_df()
        df.to_hdf(h5_path, 'image_xval_folds')


    def save(self, name):
        h5_path = os.path.join(ms2016_settings.DS_SPLITS_DIR, 'hd_seg_splits__{}.h5'.format(name))
        self.to_h5(h5_path)
        return h5_path


    @staticmethod
    def from_df(df):
        name_to_image = {img.name: img for img in get_all_images()}

        belt_names = df['Belt'].unique()

        belt_to_xv = {}
        belt_to_samples = {}

        fold_names = []
        for col_name in df.columns.values:
            if _FOLD_PATTERN.match(col_name) is not None:
                fold_names.append(col_name)

        for belt_name in belt_names:
            belt_df = df[df['Belt'] == belt_name]
            xv = training_dataset.CrossValidation([training_dataset.TrainingDataset([], [], []) for fold in fold_names],
                                                  n_samples=len(belt_df), separate_validation_and_test=True)

            for sample_name in belt_df.index:
                for fold_i, fold in enumerate(fold_names):
                    dest = belt_df.loc[sample_name][fold]
                    if dest == 'Train':
                        xv.datasets[fold_i].train.append(name_to_image[sample_name])
                    elif dest == 'Val':
                        xv.datasets[fold_i].validation.append(name_to_image[sample_name])
                    elif dest == 'Test':
                        xv.datasets[fold_i].test.append(name_to_image[sample_name])
            images_in_belt = [name_to_image[sample_name] for sample_name in belt_df.index.values]
            belt = images_in_belt[0].belt
            belt_to_xv[belt] = xv
            belt_to_samples[belt] = [name_to_image[sample_name] for sample_name in belt_df.index.values]

        return FishDataset(belt_to_xv, belt_to_samples)


    @staticmethod
    def from_h5(h5_path):
        return FishDataset.from_df(pd.read_hdf(h5_path, 'image_xval_folds'))

    @staticmethod
    def load(name):
        h5_path = os.path.join(ms2016_settings.DS_SPLITS_DIR, 'hd_seg_splits__{}.h5'.format(name))
        return FishDataset.from_h5(h5_path)



