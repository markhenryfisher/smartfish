import re

import pandas as pd

from dataset import fish_images, training_dataset

fold_pat = re.compile('Fold (\d+)')


class FishDataset(object):
    def __init__(self, belt_to_xv, belt_to_samples):
        self.belt_to_xv = belt_to_xv
        self.belt_to_samples = belt_to_samples

    def get_belt(self, belt_name):
        return self.belt_to_xv[belt_name]

    def get_fold(self, belt_name, fold_i):
        return self.belt_to_xv[belt_name].datasets[fold_i]

    def get_samples_in_belt(self, belt_name):
        return self.belt_to_samples[belt_name]

    @staticmethod
    def from_df(df):
        images = fish_images.ALL_BOAT_WARPED_IMAGES_ALL
        name_to_image = {img.name: img for img in images}

        belt_names = df['Belt'].unique()

        belt_to_xv = {}
        belt_to_samples = {}

        fold_names = []
        for col_name in df.columns.values:
            if fold_pat.match(col_name) is not None:
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
            belt_to_xv[belt_name] = xv
            belt_to_samples[belt_name] = [name_to_image[sample_name] for sample_name in belt_df.index.values]

        return FishDataset(belt_to_xv, belt_to_samples)


    @staticmethod
    def from_h5(h5_path):
        return FishDataset.from_df(pd.read_hdf(h5_path, 'image_xval_folds'))


