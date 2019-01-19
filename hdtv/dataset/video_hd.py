import os
import json

from settings import ms2016_settings



class Belt (object):
    def __init__(self, belt_name):
        self.name = belt_name

        # Video
        self.videos_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.VIDEO_DIR, self.name))

        # Calibration
        self.calib_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.CALIBRATION_DIR, self.name))
        self.__calib_cache_lens = {}
        self.__calib_cache_belt = {}
        self.__calib_cache_belt_realign = {}

        # Segmentation paths
        self.seg_input_image_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.BELT_IMAGES_DIR, self.name))
        self.seg_label_frames_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.LABEL_FRAMES_DIR, self.name))
        self.seg_labels_json_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_LABELS_JSON_DIR, self.name))
        self.seg_truth_fg_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_FG_DIR, self.name))
        self.seg_truth_mask_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_MASK_DIR, self.name))
        self.seg_truth_edges_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_EDGES_DIR, self.name))
        self.seg_truth_labels_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_LABELS_DIR, self.name))
        self.seg_deep_watershed_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_DEEP_WATERSHED_DIR, self.name))
        self.seg_maximal_ball_normal_map_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SEG_MAXBALLNORMAL_DIR, self.name))

        self.belt_stats_path = os.path.join(self.seg_input_image_dir, 'stats.npz')

        self.species_id_labels_json_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SPECIES_ID_LABELS_JSON_DIR, self.name))
        self.species_id_images_dir = ms2016_settings.ensure_dir_exists(
            os.path.join(ms2016_settings.SPECIES_ID_IMAGES_DIR, self.name))


    def __eq__(self, other):
        if isinstance(other, Belt):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash((Belt, self.name))

    def __repr__(self):
        return 'Belt({})'.format(self.name)

    def __str__(self):
        return 'Belt({})'.format(self.name)


    def lens_calibration_path(self, camera_name):
        return os.path.join(self.calib_dir, 'lens__camera_{}.json'.format(camera_name))

    def belt_realign_calibration_path(self, camera_name, src_alignment_name, dst_alignment_name):
        return os.path.join(self.calib_dir, 'beltrealign__camera_{}__align_{}__to_{}.json'.format(camera_name, src_alignment_name, dst_alignment_name))

    def belt_calibration_path(self, camera_name, alignment_name):
        return os.path.join(self.calib_dir, 'belt__camera_{}__align_{}.json'.format(camera_name, alignment_name))


    def get_lens_calibration(self, camera_name):
        key = camera_name
        try:
            return self.__calib_cache_lens[key]
        except KeyError:
            path = self.lens_calibration_path(camera_name)
            if not os.path.exists(path):
                raise RuntimeError('Could not find lens calibration file {} for camera {}'.format(path, camera_name))
            from calibration import lens
            with open(path, 'r') as f:
                calib_js = json.load(f)
            x = lens.LensCalibration.from_json(calib_js)
            self.__calib_cache_lens[key] = x
            return x

    def get_belt_calibration(self, camera_name, align_name):
        key = camera_name, align_name
        try:
            return self.__calib_cache_belt[key]
        except KeyError:
            path = self.belt_calibration_path(camera_name, align_name)
            if not os.path.exists(path):
                raise RuntimeError('Could not find belt calibration file {} for camera {} alignment {}'.format(path, camera_name, align_name))
            from calibration import belt
            with open(path, 'r') as f:
                calib_js = json.load(f)
            x = belt.BeltCalibration.from_json(calib_js)
            self.__calib_cache_belt[key] = x
            return x


    def model_path(self, dir_names, filename):
        names = list(dir_names) + [self.name, filename]
        return os.path.join(ms2016_settings.MODELS_DIR, *names)

    def output_model_path(self, dir_names, filename):
        dir_names = list(dir_names) + [self.name]
        model_dir = ms2016_settings.ensure_dir_exists(os.path.join(ms2016_settings.MODELS_DIR, *dir_names))
        return os.path.join(model_dir, filename)




BELTS = []
BELT_NAME_TO_BELT = {}
filename = None
belt = None
for filename in os.listdir(ms2016_settings.VIDEO_DIR):
    if os.path.isdir(os.path.join(ms2016_settings.VIDEO_DIR, filename)):
        belt = Belt(filename)
        BELTS.append(belt)
        BELT_NAME_TO_BELT[filename] = belt
del filename, belt




class VideoHD (object):
    def __init__(self, name, filename, path, belt):
        self.name = name
        self.filename = filename
        self.path = path
        self.motion_filename = os.path.splitext(filename)[0] + '__motion.npz'
        self.motion_path = os.path.join(belt.videos_dir, self.motion_filename)
        self.belt = belt
        self.camera_name = None
        self.align_name = None

        self.__lens_calib = None
        self.__belt_calib = None

    @property
    def lens_calibration(self):
        if self.__lens_calib is None:
            if self.camera_name is None:
                raise RuntimeError('Cannot get lens calibration for video {} as no camera name is registered; run the `register_calibration` program'.format(self.name))
            self.__lens_calib = self.belt.get_lens_calibration(self.camera_name)
        return self.__lens_calib

    @property
    def belt_calibration(self):
        if self.__belt_calib is None:
            if self.camera_name is None:
                raise RuntimeError('Cannot get belt calibration for video {} as no camera name is registered; run the `register_calibration` program'.format(self.name))
            if self.align_name is None:
                raise RuntimeError('Cannot get belt calibration for video {} as no alignment name is registered; run the `register_calibration` program'.format(self.name))
            self.__belt_calib = self.belt.get_belt_calibration(self.camera_name, self.align_name)
        return self.__belt_calib

    def compute_opencv_seek_bug_offset(self, max_search=100):
        import cv2
        import hashlib

        cap_bug = cv2.VideoCapture(self.path)
        if not cap_bug.isOpened():
            raise RuntimeError('Could not load video {}'.format(self.path))
        cap_bug.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, bug_frame_0 = cap_bug.read()
        if not success:
            raise RuntimeError('Could read frame 0 from video {}'.format(self.path))

        bug_hasher = hashlib.sha256()
        bug_hasher.update(bug_frame_0.tobytes())
        bug_frame_0_hash = bug_hasher.hexdigest()

        cap_bug.release()

        # Find the frame using non-buggy seeking
        cap_good = cv2.VideoCapture(self.path)

        if not cap_good.isOpened():
            raise RuntimeError('Could not load video {}'.format(self.path))
        for frame_i in range(max_search):
            success, good_frame = cap_good.read()
            if not success:
                break

            good_hasher = hashlib.sha256()
            good_hasher.update(good_frame.tobytes())
            good_frame_hash = good_hasher.hexdigest()

            if bug_frame_0_hash == good_frame_hash:
                return frame_i

        cap_good.release()

        return None




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
                video = VideoHD(name, filename, path, belt)
                VIDEOS.append(video)
                VIDEO_NAME_TO_VIDEO[name] = video
del filename, name, video


_CALIB_REG_INDEX_PATH = os.path.join(ms2016_settings.CALIBRATION_DIR, 'calibration_index.json')

def save_calibration_registration():
    reg = []
    for video in VIDEOS:
        entry = dict(video_name=video.name, camera_name=video.camera_name, align_name=video.align_name)
        reg.append(entry)

    with open(_CALIB_REG_INDEX_PATH, 'w') as f:
        json.dump(reg, f)


def load_calibration_registration():
    if os.path.exists(_CALIB_REG_INDEX_PATH):
        with open(_CALIB_REG_INDEX_PATH, 'r') as f:
            reg = json.load(f)
        for entry in reg:
            video_name = entry['video_name']
            if video_name.startswith('Calibration') or video_name.startswith('calibration') or \
                    video_name.endswith('__motion'):
                pass
            elif video_name in VIDEO_NAME_TO_VIDEO:
                video = VIDEO_NAME_TO_VIDEO[video_name]
                video.camera_name = entry['camera_name']
                video.align_name = entry['align_name']
            else:
                print('WARNING: could not find video named {} while loading calibration registry'.format(video_name))


load_calibration_registration()