import numpy as np

import theano, theano.tensor as T

import lasagne

from britefury_lasagne.pretrained.imagenet_vgg import VGG16Model

from batchup import data_source


class VGG16FeatureExtractor (object):
    _LAYER_NAME_TO_SIZE = {
        'conv1_1': 3,
        'conv1_2': 5,
        'pool1': 6,

        'conv2_1': 10,
        'conv2_2': 14,
        'pool2': 16,

        'conv3_1': 24,
        'conv3_2': 32,
        'conv3_3': 40,
        'pool3': 44,

        'conv4_1': 60,
        'conv4_2': 76,
        'conv4_3': 92,
        'pool4': 100,

        'conv5_1': 132,
        'conv5_2': 164,
        'conv5_3': 196,
        'pool5': 212,
    }

    def __init__(self, layer_name, batch_size=8):
        self.size = self._LAYER_NAME_TO_SIZE[layer_name]
        self.lower_pad = (self.size - 1) // 2
        self.upper_pad = self.size // 2

        # Load the VGG network
        self.__vgg16 = VGG16Model.load(input_shape=(3, self.size, self.size), last_layer_name=layer_name)

        # Get the Theano expression representing the features at the selected layer
        feats_out = lasagne.layers.get_output(self.__vgg16.network[layer_name], deterministic=True)
        # Compile Theano feature extraction function
        self._f_pred = theano.function([self.__vgg16.network['input'].input_var], feats_out)

        self.batch_size = batch_size

    # Scale to [0,255] range and mean subtract
    def vgg16_preprocess_image(self, X):
        # RGB -> BGR, then subtract vgg16 mean value
        return X[:, :, ::-1] * 255.0 - self.__vgg16.mean_value[None, None, :]

    def vgg16_preprocess_patches(self, X):
        # RGB -> BGR, then subtract vgg16 mean value
        return X[:, ::-1, :, :] * 255.0 - self.__vgg16.mean_value[None, :, None, None]

    # Patch feature extraction
    def patch_features(self, patches):
        patches = self.vgg16_preprocess_patches(patches.astype(np.float32))
        ds = data_source.ArrayDataSource([patches])
        feats, = ds.batch_map_concat(self._f_pred, batch_size=self.batch_size)
        feats = feats.reshape((feats.shape[0], -1))
        return feats

    def extract(self, img, keypoints, progress_iter_fn=None):
        img = self.vgg16_preprocess_image(img)

        img = np.pad(img, [(self.lower_pad, self.upper_pad), (self.lower_pad, self.upper_pad), (0,0)], mode='constant')
        keypoints = keypoints + self.lower_pad

        feats = []
        batches = range(0, keypoints.shape[0], self.batch_size)
        if progress_iter_fn is not None:
            batches = progress_iter_fn(batches)
        for i in batches:
            batch_kp = keypoints[i: i+self.batch_size]
            patches = np.zeros((batch_kp.shape[0], 3, self.size, self.size), dtype=np.float32)

            for j in range(batch_kp.shape[0]):
                yx = batch_kp[j]
                y, x = int(yx[0] + 0.5), int(yx[1] + 0.5)
                patches[j, ...] = img[y-self.lower_pad:y+self.upper_pad+1, x-self.lower_pad:x+self.upper_pad+1, :].transpose(2, 0, 1)

            f = self._f_pred(patches)[:,:,0,0]
            feats.append(f)

        feats = np.concatenate(feats, axis=0)
        feats = feats.reshape((feats.shape[0], -1))
        return feats
