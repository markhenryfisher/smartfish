import getpass, os, sys

import joblib


memory = None

_POSSIBLE_PROJECT_PATHS = [
    r'c:\data\fish2014',
    r'd:\data\fish2014',
    os.path.expanduser('~/fish2014'),
    r'/Volumes/GeoffSD/fish2014',
]

def get_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise ValueError('Could not find path; looked in {}'.format(paths))

PROJECT_PATH = get_existing_path(_POSSIBLE_PROJECT_PATHS)



def ensure_dir_exists(path):
    """
    Ensure that the specified directory exists; create it if it does not

    :param path: the path to check

    :return: `path`
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if os.path.exists(PROJECT_PATH):
    VIDEO_PATH = os.path.join(PROJECT_PATH, 'CCTV')
    BELT_PARAMS_PATH = os.path.join(PROJECT_PATH, 'belt_params')
    CACHE_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'cache'))
    LABEL_IMAGE_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'label_images'))
    OBJECT_LABEL_IMAGE_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'object_label_images'))

    SEGMENTATION_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'segmentation_test_data'))
    SEGMENTATION_INPUT_PATH = ensure_dir_exists(os.path.join(SEGMENTATION_PATH, 'in'))
    SEGMENTATION_MASK_PATH = ensure_dir_exists(os.path.join(SEGMENTATION_PATH, 'mask'))
    SEGMENTATION_TRUTH_PATH = ensure_dir_exists(os.path.join(SEGMENTATION_PATH, 'truth'))
    SEGMENTATION_MODELS_PATH = ensure_dir_exists(os.path.join(SEGMENTATION_PATH, 'models'))

    SEPARATION_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'separation_tests'))
    SEPARATION_MODELS_PATH = ensure_dir_exists(os.path.join(SEPARATION_PATH, 'models'))
    SEPARATION_EDGE_HEAT_PATH = ensure_dir_exists(os.path.join(SEPARATION_PATH, 'out_edge_heat'))

    LOCAL_DIST_SEPARATION_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'local_dist_separation_tests'))
    LOCAL_DIST_SEPARATION_MODELS_PATH = ensure_dir_exists(os.path.join(LOCAL_DIST_SEPARATION_PATH, 'models'))
    LOCAL_DIST_SEPARATION_EDGE_HEAT_PATH = ensure_dir_exists(os.path.join(LOCAL_DIST_SEPARATION_PATH, 'out_edge_heat'))

    WARP_SEG_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'warpseg_test_data'))
    WARP_SEG_INPUT_PATH = ensure_dir_exists(os.path.join(WARP_SEG_PATH, 'in'))
    WARP_SEG_TRUTH_PATH = ensure_dir_exists(os.path.join(WARP_SEG_PATH, 'truth'))
    WARP_SEG_MASK_PATH = ensure_dir_exists(os.path.join(WARP_SEG_PATH, 'mask'))
    WARP_SEG_MODELS_PATH = ensure_dir_exists(os.path.join(WARP_SEG_PATH, 'models'))

    ALLBOATS_WARP_SEG_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'allboat_warpseg_test_data'))
    ALLBOATS_WARP_SEG_INPUT_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'in'))
    ALLBOATS_WARP_SEG_TRUTH_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'truth'))
    ALLBOATS_WARP_SEG_MULTICLASS_TRUTH_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'multiclass_truth'))
    ALLBOATS_WARP_SEG_MASK_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'mask'))
    ALLBOATS_WARP_SEG_EDGES_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'edges'))
    ALLBOATS_WARP_SEG_GAUSSIAN_EDGES_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'gaussian_edges'))
    ALLBOATS_WARP_SEG_HARD_EDGES_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'hard_edges'))
    ALLBOATS_WARP_SEG_ALL_HARD_EDGES_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'hard_edges_all'))
    ALLBOATS_WARP_SEG_DIRFREQ_EDGES_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'edges_dirfreq'))
    ALLBOATS_WARP_SEG_LABELS_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'labels'))
    ALLBOATS_WARP_SEG_ALL_LABELS_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'labels_all'))
    ALLBOATS_WARP_SEG_OUT_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'out'))
    ALLBOATS_WARP_SEG_MODELS_PATH = ensure_dir_exists(os.path.join(ALLBOATS_WARP_SEG_PATH, 'models'))

    DATA_AUG_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'data_aug'))

    VIDEO_OUTPUT_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'video_out'))
    VIDEO_OUTPUT_201702_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'video_out_201702'))

    VIDEO_MODELS_PATH = ensure_dir_exists(os.path.join(VIDEO_OUTPUT_PATH, 'models'))
    VIDEO_FRAMES_PATH = ensure_dir_exists(os.path.join(VIDEO_OUTPUT_PATH, 'frames'))

    EXPERIMENT_RESULTS_PATH = ensure_dir_exists(os.path.join(PROJECT_PATH, 'exp_results'))


    def video_out_model_path(video_name, *model_name_parts):
        parts = [VIDEO_MODELS_PATH] + list(model_name_parts) + [video_name]
        return os.path.join(*parts)

    def video_out_model_path_for_writing(video_name, *model_name_parts):
        parts = [VIDEO_MODELS_PATH] + list(model_name_parts) + [video_name]
        dir_path = ensure_dir_exists(os.path.join(*parts[:-1]))
        return os.path.join(dir_path, parts[-1])


    def video_out_frames_dir_path(video_name, *name_parts):
        parts = [VIDEO_FRAMES_PATH] + list(name_parts) + [video_name]
        return os.path.join(*parts)

    def video_out_frames_dir_path_for_writing(video_name, *name_parts):
        return ensure_dir_exists(video_out_frames_dir_path(video_name, *name_parts))


    # Create the joblib memory cache
    memory = joblib.Memory(cachedir=CACHE_PATH, verbose=0)

else:
    raise RuntimeError('Could not initialise Marine Scotland, the project path \'{0}\' does not exist'.format(PROJECT_PATH))


