# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import os
# import sys
# import datetime
import gzip

import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from PIL import Image

from .download import maybe_download
from .download import get_data_dir

from .workflow import pprint
from .workflow import check_dir
from .workflow import _get_save_directory


RANDOM_STATE = 42

def _load_mnist_images(filename):
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: [batch_size, image_width, image_height, channels]
    data = data.reshape(-1, 28, 28, 1)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data
    # return data / np.float32(256)

def _load_mnist_labels(filename):
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

def load_data():
    """
    TODO : doc
    """
    source_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    fname_train_images = 'train-images-idx3-ubyte.gz'
    fname_train_labels = 'train-labels-idx1-ubyte.gz'
    fname_test_images = 't10k-images-idx3-ubyte.gz'
    fname_test_labels = 't10k-labels-idx1-ubyte.gz'
    data_dir = get_data_dir()
    maybe_download(os.path.join(data_dir, 'fashion'+fname_train_images), source_url+fname_train_images)
    maybe_download(os.path.join(data_dir, 'fashion'+fname_train_labels), source_url+fname_train_labels)
    maybe_download(os.path.join(data_dir, 'fashion'+fname_test_images), source_url+fname_test_images)
    maybe_download(os.path.join(data_dir, 'fashion'+fname_test_labels), source_url+fname_test_labels)

    X_train = _load_mnist_images(os.path.join(data_dir, 'fashion'+fname_train_images))
    y_train = _load_mnist_labels(os.path.join(data_dir, 'fashion'+fname_train_labels))
    X_test = _load_mnist_images(os.path.join(data_dir, 'fashion'+fname_test_images))
    y_test = _load_mnist_labels(os.path.join(data_dir, 'fashion'+fname_test_labels))
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    return X, y


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


def tangent(x, alpha=1):
    """ The approximate formula to get the tangent. """
    x_plus = skew(x, z=alpha) / 255
    x_minus = skew(x, z=-alpha) / 255
    return ( x_plus - x_minus ) / ( 2 * alpha )


def preprocessing(X, y, w, training=True):
    X = X.reshape(-1, 28*28) / 255
    return X, y, w

def get_cv_iter(X, y):
    cv = ShuffleSplit(n_splits=12, test_size=0.2, random_state=RANDOM_STATE)
    cv_iter = list(cv.split(X, y))
    return cv_iter


def get_save_directory():
    dir = os.path.join( _get_save_directory(), 'fashion_mnist')
    return check_dir(dir)

def train_submission(model, X, y):
    cv_iter = get_cv_iter(X, y)
    n_cv = len(cv_iter)
    save_directory = get_save_directory()
    for i, (idx_dev, idx_valid) in enumerate(cv_iter):
        X_train = X[idx_dev]
        y_train = y[idx_dev]
        
        pprint('training {}/{}...'.format(i+1, n_cv))
        model.fit(X_train, y_train)

        pprint('saving model {}/{}...'.format(i+1, n_cv))
        model_name = '{}-{}'.format(model.get_name(), i)
        
        path = os.path.join(save_directory, model_name)
        check_dir(path)
        
        model.save(path)
    return None


def test_submission(models, X, y, z_list=(-45, 0, +45)):
    cv_iter = get_cv_iter(X, y)
    n_cv = len(cv_iter)
    df_list = []
    for i, (idx_dev, idx_valid) in enumerate(cv_iter):
        X_test = X[idx_valid]
        y_test = y[idx_valid]
        res = []
        model = models[i]
        pprint('testing model {}/{}'.format(i+1, n_cv))
        for z in z_list:
            X_t = skew(X_test, z=z)
            pred = model.predict(X_t)
            acc = accuracy_score(y_test, pred)
            res.append((z, acc))
        df = pd.DataFrame(res, columns=['z', 'accuracy'])
        df_list.append(df)
    pprint('Done.')
    return df_list
