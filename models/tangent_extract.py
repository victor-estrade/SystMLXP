# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np


class TangentExtractor(object):
    def __init__(self, skewing_function, alpha=1e-2):
        super().__init__()
        self.skewing_function = skewing_function
        self.alpha = alpha

    def compute_tangent(self, X):
        """ The approximate formula to get the tangent. """
        X_plus = self.skewing_function(X, z=self.alpha)
        X_minus = self.skewing_function(X, z=-self.alpha)
        return ( X_plus - X_minus ) / ( 2 * self.alpha )

class TangentComputer(object):
    """ For 2D rotation """
    def __init__(self):
        super().__init__()

    def compute_tangent(self, X):
        """ The real formula to get the tangent. """
        X_2 = X*X
        rho = np.sqrt(X_2[:,0]+X_2[:,1])
        theta = np.arctan2(X[:, 1], X[:, 0])
        theta = theta
        X_2[:, 0] = -rho*np.sin(theta)
        X_2[:, 1] = rho*np.cos(theta)
        return X_2