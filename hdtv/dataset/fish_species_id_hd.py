import os
import io
import json
import numpy as np
import cv2
from dataset import video_hd
from skimage.util import img_as_float
from sklearn.model_selection import GroupShuffleSplit
from dataset.fish_seg_hd import as_grey

CLSI_NONE = -1
CLSI_UNKNOWN = -1
CLSI_NOT_FISH = 0
CLSI_FISH_VALID_COD = 1
CLSI_FISH_VALID_HADDOCK = 2
CLSI_FISH_VALID_WHITING = 3
CLSI_FISH_VALID_SAITHE = 4
CLSI_FISH_VALID_HAKE = 5
CLSI_FISH_VALID_MONK = 6
CLSI_FISH_VALID_MISC = 7
CLSI_FISH_VALID_SMALL = 8
CLSI_FISH_VALID_PARTIAL = 9
CLSI_FISH_MULTIPLE = -1
CLSI_FISH_UNKNOWN = -2


FISH_VALID_CLS_INDICES = {
    CLSI_FISH_VALID_COD, CLSI_FISH_VALID_HADDOCK, CLSI_FISH_VALID_WHITING, CLSI_FISH_VALID_SAITHE, CLSI_FISH_VALID_HAKE, CLSI_FISH_VALID_MONK,
    CLSI_FISH_VALID_MISC, CLSI_FISH_VALID_SMALL, CLSI_FISH_VALID_PARTIAL
}

FISH_KNOWN_SPECIES_INDICES = {
    CLSI_FISH_VALID_COD, CLSI_FISH_VALID_HADDOCK, CLSI_FISH_VALID_WHITING, CLSI_FISH_VALID_SAITHE, CLSI_FISH_VALID_HAKE, CLSI_FISH_VALID_MONK,
}



_SPECIES_CLS_NAME_TO_INDEX = {
    None: CLSI_NONE,
    'unknown': CLSI_UNKNOWN,
    'not_fish': CLSI_NOT_FISH,
    'fish_cod': CLSI_FISH_VALID_COD,
    'fish_haddock': CLSI_FISH_VALID_HADDOCK,
    'fish_whiting': CLSI_FISH_VALID_WHITING,
    'fish_saithe': CLSI_FISH_VALID_SAITHE,
    'fish_hake': CLSI_FISH_VALID_HAKE,
    'fish_monk': CLSI_FISH_VALID_MONK,
    'fish_misc': CLSI_FISH_VALID_MISC,
    'fish_small': CLSI_FISH_VALID_SMALL,
    'fish_partial': CLSI_FISH_VALID_PARTIAL,
    'fish_multiple': CLSI_FISH_MULTIPLE,
    'fish_unknown': CLSI_FISH_UNKNOWN,
}

_SPECIES_CLS_INDEX_TO_NAME = {ndx: name for name, ndx in _SPECIES_CLS_NAME_TO_INDEX.items()}





def _n_bytes_for_n_labels(n_labels):
    n_bytes = n_labels // 8
    if (n_labels % 8) != 0:
        n_bytes += 1
    return n_bytes


def new_species_id_label_image(height, width, n_labels):
    n_bytes = _n_bytes_for_n_labels(n_labels)
    return np.zeros((height, width, n_bytes), dtype=np.uint8)


def add_label_mask(label_image, mask, label_index):
    if label_image.shape[:2] != mask.shape:
        raise ValueError('shape mismatch; label_image.shape={}, mask.shape={}'.format(label_image.shape, mask.shape))
    byte_ndx = label_index // 8
    bit_ndx = label_index % 8
    mask_val = ((mask != 0) * (1 << bit_ndx)).astype(np.uint8)
    label_image[:, :, byte_ndx] |= mask_val


def get_label_mask(label_image, label_index):
    byte_ndx = label_index // 8
    bit_ndx = label_index % 8
    byte_img = label_image[:, :, byte_ndx]
    bit_mask = 1 << bit_ndx
    return (byte_img & bit_mask) >> bit_ndx




class ObjectImage (object):
    def __init__(self, belt, video, object_index, y):
        self.belt = belt
        self.video = video
        self.object_index = object_index
        self.name = '{}__obj_{:06d}'.format(video.name, object_index)
        self.object_image_filename = '{}_object.png'.format(self.name)
        self.mask_image_filename = '{}_mask.png'.format(self.name)
        self.object_image_path = os.path.join(belt.species_id_images_dir, self.object_image_filename)
        self.mask_image_path = os.path.join(belt.species_id_images_dir, self.mask_image_filename)
        self.y = y

    def read_object_image_u8(self):
        return cv2.imread(self.object_image_path)[:, :, ::-1]

    def read_mask_image(self):
        img = cv2.imread(self.mask_image_path)
        return as_grey(img)


    def to_json(self):
        return dict(belt_name=self.belt.name, video_name=self.video.name, object_index=self.object_index, y=self.y)

    @staticmethod
    def from_json(js):
        belt = video_hd.BELT_NAME_TO_BELT[js['belt_name']]
        video = video_hd.VIDEO_NAME_TO_VIDEO[js['video_name']]
        assert video.belt is belt
        return ObjectImage(belt, video, js['object_index'], js['y'])


    @staticmethod
    def from_species_id_ground_truth_json(gt):
        if isinstance(gt, str):
            # Filename
            js = json.load(open(gt, 'r'))
        elif isinstance(gt, io.IOBase):
            # File
            js = json.load(gt)
        elif isinstance(gt, list):
            # JSON
            js = gt
        else:
            raise TypeError('gt should be a path (str), a file or a list (JSON), not a {}'.format(type(gt)))

        return [ObjectImage.from_json(entry) for entry in js]


    @staticmethod
    def belt_species_id_ground_truth_path(belt):
        return os.path.join(belt.species_id_images_dir, 'ground_truth.json')


    @staticmethod
    def for_belt(belt):
        gt_path = ObjectImage.belt_species_id_ground_truth_path(belt)
        if os.path.exists(gt_path):
            return ObjectImage.from_species_id_ground_truth_json(gt_path)
        else:
            return []


    @staticmethod
    def split_by_video(objs, test_size=0.25, random_state=12345):
        video_to_index = {}
        y = np.array([obj.y for obj in objs])
        video_indices = np.array([video_to_index.setdefault(obj.video.name, len(video_to_index)) for obj in objs])
        split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_ndx, test_ndx = next(split.split(y, y, video_indices))
        train_video_names = {objs[i].video.name for i in train_ndx}
        test_video_names = {objs[i].video.name for i in test_ndx}
        return train_ndx, test_ndx, train_video_names, test_video_names



class AbstractObjectImageArrayAccessor(object):
    def __init__(self, objects):
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def read_object_image(self, img):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.objects[index].read_object_image_u8()
        elif isinstance(index, (slice, np.ndarray)):
            indices = np.arange(len(self))[index]
            images = [self.read_object_image(self.objects[i]) for i in indices]
            return np.concatenate([img[None, ...] for img in images], axis=0)
        else:
            raise TypeError

class ObjectImageU8ArrayAccessor(AbstractObjectImageArrayAccessor):
    def read_object_image(self, img):
        return img.read_object_image_u8().transpose(2, 0, 1)

class ObjectMaskArrayAccessor(AbstractObjectImageArrayAccessor):
    def read_object_image(self, img):
        return img.read_mask_image()[None, ...]
