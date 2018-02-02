# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

class DataAugmenter(object):
    def __init__(self):
        super().__init__()
    
    def augment(self, X, y, sample_weight=None):
        raise NotImplementedError
    
    def __call__(self, X, y, sample_weight=None):
        return self.augment(X, y, sample_weight=None)
    
    def sample_z(self, size):
        raise NotImplementedError

class NormalDataAugmenter(DataAugmenter):
    def __init__(self, skewing_function, width=1, center=0, n_augment=2):
        super().__init__()
        self.skewing_function = skewing_function
        self.width = width
        self.center = center
        self.n_augment = n_augment
    
    def augment(self, X, y, sample_weight=None):
        z_list = [self.sample_z( size=X.shape[0] ) for _ in range(self.n_augment)]
        X = np.concatenate( [X,] + [ self.skewing_function(X, z) for z in z_list ], axis=0)
        y = np.concatenate( [y,] + [y for _ in range(self.n_augment) ], axis=0)
        if sample_weight is not None:
            W = np.concatenate( [sample_weight,] + [sample_weight for _ in range(self.n_augment) ], axis=0)
            return X, y, W
        return X, y, sample_weight

    def sample_z(self, size):
        z = np.random.normal( loc=self.center, scale=self.width, size=size )
        return z