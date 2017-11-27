# coding : utf-8
import numpy as np

from collections import Generator

def assert_arrays_have_same_shape(*arrays):
    length = arrays[0].shape[0]
    # Assert that every array have the same 1st dimension length:
    for i, arr in enumerate(arrays):
        assert arr.shape[0] == length, "Every array should have the same shape: " \
                        " array {} length = {}  but expected length = {} ".format(i+1, arr.shape[0], length)

def uniform_sampling(*args, batch_size=None):
    """
    Return a generator taking 'batch_size' random samples from given arrays.
    """
    if batch_size is None:
        raise ValueError('batch_size should not be None !')
    if len(args) == 0:
        raise ValueError('minibatching must take at least one array')
    assert_arrays_have_same_shape(*args)

    while(True):
        idx = np.random.choice(length, size=batch_size)
        yield tuple(arr[idx] for arr in args)


def epoch_shuffle(*args, batch_size=None, shuffle=True):
    """
    Return a generator taking 'batch_size' random samples from X and y.
    """
    if batch_size is None:
        raise ValueError('batch_size should not be None !')
    if len(args) == 0:
        raise ValueError('minibatching must take at least one array')
    assert_arrays_have_same_shape(*args)

    size = args[0].shape[0]
    assert size > batch_size, 'batch_size should be smaller than the number of samples in the given arrays'
    
    while(True):
        if shuffle:
            indices = np.arange(size)
            np.random.shuffle(indices)
        for start_idx in range(0, size - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield tuple(arr[excerpt] for arr in args)



class OneEpoch(Generator):
    def __init__(self, *arrays, batch_size=None):
        """
        Return a generator taking 'batch_size' samples from X and y.
        """
        if batch_size is None:
            raise ValueError('batch_size should not be None !')
        if len(arrays) == 0:
            raise ValueError('minibatching must take at least one array')
        assert_arrays_have_same_shape(*arrays)

        size = arrays[0].shape[0]
        assert size > batch_size, 'batch_size should be smaller than the number of samples in the given arrays'
        
        self.arrays = arrays
        self.start_idx = 0
        self.batch_size = batch_size
        self.size = size
        self.step = 0
        self.max_start_idx = size - batch_size + 1
        self.yielded = 0

    def send(self, ignored_arg):
        if self.start_idx >= self.size:
            self.throw()
        excerpt = slice(self.start_idx, self.start_idx + self.batch_size)
        self.start_idx += self.batch_size
        self.step += 1
        self.yielded += self.batch_size if self.start_idx < self.size else self.batch_size - self.start_idx + self.size
        if len(self.arrays) == 1:
            return self.arrays[0][excerpt]
        else:
            return tuple(arr[excerpt] for arr in self.arrays)
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


class Epoch(Generator):
    def __init__(self, *arrays, batch_size=None):
        """
        Return a generator taking 'batch_size' samples from X and y.
        """
        if batch_size is None:
            raise ValueError('batch_size should not be None !')
        if len(arrays) == 0:
            raise ValueError('minibatching must take at least one array')
        assert_arrays_have_same_shape(*arrays)

        size = arrays[0].shape[0]
        assert size > batch_size, 'batch_size should be smaller than the number of samples in the given arrays'
        
        self.arrays = arrays
        self.start_idx = 0
        self.batch_size = batch_size
        self.size = size
        self.step = 0
        self.max_start_idx = size - batch_size + 1
        self.epoch = 0
        self.yielded = 0

    def send(self, ignored_arg):
        if self.start_idx >= self.size:
            self.start_idx = 0
            self.epoch += 1
        excerpt = slice(self.start_idx, self.start_idx + self.batch_size)
        self.start_idx += self.batch_size
        self.step += 1
        self.yielded += self.batch_size if self.start_idx < self.size else self.batch_size - self.start_idx + self.size
        if len(self.arrays) == 1:
            return self.arrays[0][excerpt]
        else:
            return tuple(arr[excerpt] for arr in self.arrays)
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


class EpochShuffle(Generator):
    def __init__(self, *arrays, batch_size=None):
        """
        Return a generator taking 'batch_size' random samples from X and y.
        """
        if batch_size is None:
            raise ValueError('batch_size should not be None !')
        if len(arrays) == 0:
            raise ValueError('minibatching must take at least one array')
        assert_arrays_have_same_shape(*arrays)

        size = arrays[0].shape[0]
        assert size > batch_size, 'batch_size should be smaller than the number of samples in the given arrays'
        
        self.arrays = arrays
        self.start_idx = 0
        self.batch_size = batch_size
        self.size = size
        self.step = 0
        self.max_start_idx = size - batch_size + 1
        self.epoch = 0
        self.yielded = 0
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)

    def send(self, ignored_arg):
        if self.start_idx >= self.size:
            self.start_idx = 0
            self.epoch += 1
            np.random.shuffle(self.indices)
            
        excerpt = self.indices[self.start_idx:self.start_idx + self.batch_size]
        self.start_idx += self.batch_size
        self.step += 1
        self.yielded += self.batch_size if self.start_idx < self.size else self.batch_size - self.start_idx + self.size
        if len(self.arrays) == 1:
            return self.arrays[0][excerpt]
        else:
            return tuple(arr[excerpt] for arr in self.arrays)
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


class OneEpochShuffle(Generator):
    def __init__(self, *arrays, batch_size=None):
        """
        Return a generator taking 'batch_size' samples from X and y.
        """
        if batch_size is None:
            raise ValueError('batch_size should not be None !')
        if len(arrays) == 0:
            raise ValueError('minibatching must take at least one array')
        assert_arrays_have_same_shape(*arrays)

        size = arrays[0].shape[0]
        assert size > batch_size, 'batch_size should be smaller than the number of samples in the given arrays'
        
        self.arrays = arrays
        self.start_idx = 0
        self.batch_size = batch_size
        self.size = size
        self.step = 0
        self.max_start_idx = size - batch_size + 1
        self.yielded = 0
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)

    def send(self, ignored_arg):
        if self.start_idx >= self.size:
            self.throw()
        excerpt = self.indices[self.start_idx:self.start_idx + self.batch_size]
        self.start_idx += self.batch_size
        self.step += 1
        self.yielded += self.batch_size if self.start_idx < self.size else self.batch_size - self.start_idx + self.size
        if len(self.arrays) == 1:
            return self.arrays[0][excerpt]
        else:
            return tuple(arr[excerpt] for arr in self.arrays)
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
