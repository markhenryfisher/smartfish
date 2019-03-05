import numpy as np
import six
from skimage.util import img_as_float

class ImageLoader (object):
    def __init__(self, images, load_as_tensor_fn):
        self.images = images
        self.load_as_tensor_fn = load_as_tensor_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        imgs = []
        if isinstance(item, six.integer_types):
            return self.load_as_tensor_fn(self.images[item])
        elif isinstance(item, slice):
            imgs = self.images[item]
        elif isinstance(item, np.ndarray):
            imgs = [self.images[i] for i in item]
        return [self.load_as_tensor_fn(img) for img in imgs]


class ImageOp (object):
    def __call__(self, batch):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))


class Compose (ImageOp):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, imgs):
        for op in self.ops:
            imgs = op(imgs)[0]
        return (imgs,)


class StandardiseTensor (ImageOp):
    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, X):
        if self.std is not None:
            X = (X - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        else:
            X = X - self.mean[None, :, None, None]
        return (X.astype(np.float32),)


class ImagesToTensor (ImageOp):
    def __call__(self, imgs):
        xs = []
        for img in imgs:
            # img = img_as_float(img).astype(np.float32)
            img = img.astype(np.float32)
            img = img.transpose(2, 0, 1)[None, ...]
            xs.append(img)
        return (np.concatenate(xs, axis=0).astype(np.float32),)


