import six
import numpy as np
from sklearn.model_selection import KFold


def coerce_rng(rng):
    if rng is None:
        return np.random.RandomState(12345)
    elif isinstance(rng, six.integer_types):
        return np.random.RandomState(rng)
    elif isinstance(rng, np.random.RandomState):
        return rng
    else:
        raise TypeError



class TrainingDataset (object):
    """
    A training dataset that has been split for training, validation and test

    Access the training, test and validation sets using the `train`, `test` and `validation` attributes.
    Note that `test` will be None if separate validation and test sets are not desired
    """
    def __init__(self, train, validation, test=None):
        self.train = train
        self.test = test
        self.validation = validation


    def shuffled(self, shuffle_rng=None):
        shuffle_rng = coerce_rng(shuffle_rng)

        train = self.train[:]
        shuffle_rng.shuffle(train)

        validation = self.validation[:]
        shuffle_rng.shuffle(validation)

        if self.test is not None:
            test = self.test[:]
            shuffle_rng.shuffle(test)
        else:
            test = None

        return TrainingDataset(train, validation, test)


    def copy(self):
        test = self.test[:] if self.test is not None else None
        return TrainingDataset(self.train[:], self.validation[:], test)


    def extend(self, d):
        if self.test is not None and d.test is None:
            raise ValueError('Cannot join TrainingDataset; self.test is not None, d.test is None')
        if self.test is None and d.test is not None:
            raise ValueError('Cannot join TrainingDataset; self.test is None, d.test is not None')
        self.train.extend(d.train)
        self.validation.extend(d.validation)
        if self.test is not None:
            self.test.extend(d.test)


    @staticmethod
    def tv_split(samples, val_size=0.1, rng=None):
        # Ensure RNG is usable
        rng = coerce_rng(rng)

        N = len(samples)

        # Generate and shuffle indices
        indices = np.arange(N)
        rng.shuffle(indices)

        if isinstance(val_size, int):
            n_val = val_size
        elif isinstance(val_size, float):
            n_val = int(N * val_size + 0.5)
        else:
            raise TypeError('val_size must be an int - an absolute size - or a float - size as a fraction')

        train = [samples[x]   for x in indices[:-n_val]]
        validation = [samples[x]   for x in indices[-n_val:]]

        return TrainingDataset(train, validation)


    @staticmethod
    def tvt_split(samples, val_size=0.1, test_size=0.1, rng=None):
        # Ensure RNG is usable
        rng = coerce_rng(rng)

        N = len(samples)

        # Generate and shuffle indices
        indices = np.arange(N)
        rng.shuffle(indices)

        if isinstance(val_size, int):
            n_val = val_size
        elif isinstance(val_size, float):
            n_val = int(N * val_size + 0.5)
        else:
            raise TypeError('val_size must be an int - an absolute size - or a float - size as a fraction')

        if isinstance(test_size, int):
            n_test = test_size
        elif isinstance(test_size, float):
            n_test = int(N * test_size + 0.5)
        else:
            raise TypeError('test_size must be an int - an absolute size - or a float - size as a fraction')

        n_val_test = n_val + n_test
        train = [samples[x]   for x in indices[:-n_val_test]]
        validation = [samples[x]   for x in indices[-n_val_test:-n_test]]
        test = [samples[x]   for x in indices[-n_test:]]

        return TrainingDataset(train, validation, test)






class CrossValidation (object):
    def __init__(self, datasets, n_samples, separate_validation_and_test=True):
        self.n_folds = len(datasets)
        self.separate_validation_and_test = separate_validation_and_test
        self.N = n_samples
        self.datasets = datasets


    @staticmethod
    def from_samples(samples, n_folds=5, separate_validation_and_test=True, rng=None):
        # Ensure RNG is usable
        rng = coerce_rng(rng)

        N = len(samples)
        datasets = []

        if N > 0:
            kf_train_valtest = KFold(n_splits=n_folds, shuffle=True, random_state=rng)
            for train_indices, valtest_indices in kf_train_valtest.split(np.arange(N)):
                train = list(np.array(samples)[train_indices])
                if separate_validation_and_test:
                    kf_val_test = KFold(n_splits=2, shuffle=True, random_state=rng)
                    for test_sub, val_sub in kf_val_test.split(valtest_indices):
                        val_indices = valtest_indices[val_sub]
                        test_indices = valtest_indices[test_sub]
                        break
                    validation = list(np.array(samples)[val_indices])
                    test = list(np.array(samples)[test_indices])
                else:
                    validation = list(np.array(samples)[valtest_indices])
                    test = None

                dataset = TrainingDataset(train, validation, test)
                datasets.append(dataset)
        else:
            for i in range(n_folds):
                dataset = TrainingDataset([], [], [])
                datasets.append(dataset)

        return CrossValidation(datasets, N, separate_validation_and_test)



    def extend(self, xv):
        if xv.n_folds != self.n_folds:
            raise ValueError('Cannot join CrossValidation: self.n_folds={0}, vx.n_folds={1}'.format(self.n_folds, xv.n_folds))
        if xv.separate_validation_and_test != self.separate_validation_and_test:
            raise ValueError('Cannot join CrossValidation: self.separate_validation_and_test={0}, vx.separate_validation_and_test={1}'.format(
                    self.separate_validation_and_test, xv.separate_validation_and_test))
        self.N += xv.N
        for sds, xvds in zip(self.datasets, xv.datasets):
            sds.extend(xvds)


class TensorData (object):
    def __init__(self, data):
        self.data = data

    def iterate_minibatches(self, batchsize, shuffle=False):
        N = self.data[0].shape[0]
        for d1 in self.data[1:]:
            assert d1.shape[0] == N
        if shuffle:
            indices = np.arange(N)
            np.random.shuffle(indices)
        for start_idx in range(0, N - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield [d[excerpt] for d in self.data]




from unittest import TestCase

class Test_CrossValidation (TestCase):
    def _test_no_intersection(self, a, b):
        """
        Ensure that there is no intersection between two sets A and B
        :param a: set A (will be coerced into a set)
        :param b: set B (will be coerced into a set)
        """
        a = set(a)
        b = set(b)
        self.assertEquals(set(), a.intersection(b))

    def _validate_dataset(self, dataset, samples):
        """
        Ensure that there is no intersection between the train, and validation (if present) sets
        :param dataset: dataset to validate
        :param samples: the original samples that were split
        """
        self._test_no_intersection(dataset.train, dataset.validation)
        xs = list(dataset.train) + dataset.validation
        if dataset.test is not None:
            self._test_no_intersection(dataset.train, dataset.test)
            self._test_no_intersection(dataset.validation, dataset.test)
            xs.extend(dataset.test)
        self.assertEqual(set(xs), set(samples))


    def _validate_cross_validation(self, xv, samples):
        """
        Validate the train/test/validation separation within folds

        :param xv: a CrossValidation instance
        :param samples: the original samples that were split
        """
        for dataset in xv.datasets:
            self._validate_dataset(dataset, samples)
        for i in range(xv.n_folds):
            f0 = xv.datasets[i]
            for j in range(xv.n_folds):
                if j != i:
                    f1 = xv.datasets[j]
                    self._test_no_intersection(f0.validation, f1.validation)
                    if xv.separate_validation_and_test:
                        self._test_no_intersection(f0.test, f1.test)
                        self._test_no_intersection(f0.validation, f1.test)
                        self._test_no_intersection(f0.test, f1.validation)

    def test_xv(self):
        samples = [str(x)   for x in range(47)]

        self._validate_cross_validation(CrossValidation.from_samples(samples, n_folds=5, rng=12345,
                                                                     separate_validation_and_test=True), samples)
        self._validate_cross_validation(CrossValidation.from_samples(samples, n_folds=5, rng=12345,
                                                                     separate_validation_and_test=False), samples)


    def test_tv_split(self):
        samples = [str(x)   for x in range(47)]
        ds = TrainingDataset.tv_split(samples, val_size=0.1, rng=12345)

        self._validate_dataset(ds, samples)

    def test_tvt_split(self):
        samples = [str(x)   for x in range(47)]
        ds = TrainingDataset.tvt_split(samples, val_size=0.1, test_size=0.1, rng=12345)

        self._validate_dataset(ds, samples)