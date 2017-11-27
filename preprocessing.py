# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from PIL import Image

def skew(X, z):
    if not hasattr(z, "__len__"):
        z = np.ones(X.shape[0]) * z
    X_rotated = np.empty_like(X)
    for i in range(X.shape[0]):
        x = X[i].reshape(28,28)
        img = Image.fromarray(x)
        img = img.rotate(z[i], resample=Image.BICUBIC)
        x = np.array(img)
        X_rotated[i] = x.reshape(28,28,1)
    return X_rotated