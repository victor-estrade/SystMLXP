# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

class ZComputer(object):
    def __init__(self):
        super().__init__()

    def compute_z(self, X):
        x = X[:, 0]
        y = X[:, 1]
        z = np.arctan2(y, x)
        return z
    