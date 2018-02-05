# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


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

