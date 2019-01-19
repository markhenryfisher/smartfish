import os, glob

_DATA_PATHS = [
    r'E:\data\fish2016',
    r'C:\data\fish2016',
    r'/mnt/disk0/gfrench/fish2016',
    r'C:\fish\data\fish2016', # temp fix
]

_SCRATCH_PATHS = [
    r'F:\data\fish2016',
    r'H:\data\fish2016',
    r'/mnt/disk1/gfrench/fish2016',
    r'C:\scratch\data\fish2016', # temp fix
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
VIDEO_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'CCTV-HD'))
LABEL_FRAMES_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'label_frames'))
BELT_IMAGES_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'belt_images'))

SEG_LABELS_JSON_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_labels_json'))
SEG_FG_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_truth_fg'))
SEG_MASK_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_truth_mask'))
SEG_EDGES_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_truth_edges'))
SEG_LABELS_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_truth_labels'))
SEG_DEEP_WATERSHED_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_truth_deepwatershed'))
SEG_MAXBALLNORMAL_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'seg_truth_maxballnormal'))

SPECIES_ID_LABELS_JSON_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'species_id_labels_json'))
SPECIES_ID_IMAGES_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'species_id_images'))

DS_SPLITS_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'dataset_splits'))
MODELS_DIR = ensure_dir_exists(os.path.join(SCRATCH_PATH, 'models'))
PREDICTIONS_DIR = ensure_dir_exists(os.path.join(SCRATCH_PATH, 'predictions'))
RESULTS_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'results'))
PUB_RESULTS_DIR = ensure_dir_exists(os.path.join(DATA_PATH, 'pub_results'))



def video_path(belt_name, video_filename):
    return os.path.join(VIDEO_DIR, belt_name, video_filename)

def video_path_to_belt_and_video_name(video_path):
    rel = os.path.relpath(video_path, VIDEO_DIR)
    if rel.startswith('..'):
        raise ValueError('Video at {} is not within video directory {}'.format(video_path, VIDEO_DIR))
    belt_dir, video_filename = os.path.split(rel)
    return belt_dir, video_filename

def motion_path_for_video(video_path):
    video_dir, filename = os.path.split(video_path)
    motion_filename = os.path.splitext(filename)[0] + '__motion.npz'
    return os.path.join(video_dir, motion_filename)


def find_lens_and_belt_calibration_paths(video_path_or_dir):
    if not os.path.exists(video_path_or_dir):
        raise ValueError('Path {} does not exist'.format(video_path_or_dir))

    if os.path.isfile(video_path_or_dir):
        video_dir = os.path.dirname(video_path_or_dir)
    else:
        video_dir = video_path_or_dir

    filenames = os.listdir(video_dir)
    lens = None
    belt = None
    for f in filenames:
        p = os.path.join(video_dir, f)
        if os.path.isfile(p):
            if f.startswith('lenscalibration ') and os.path.splitext(p)[1].lower() == '.json':
                if lens is None:
                    lens = p
            elif f.startswith('beltcalibration ') and os.path.splitext(p)[1].lower() == '.json':
                if belt is None:
                    belt = p

    if lens is None:
        raise ValueError('Could not find lens calibration for {}'.format(video_path_or_dir))
    if belt is None:
        raise ValueError('Could not find belt calibration for {}'.format(video_path_or_dir))

    return lens, belt
