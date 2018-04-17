# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from .neural_network_model import NeuralNetModel
from ..classifier_filter import ClassifierFilter


class CascadeNeuralNetModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_steps=5000, batch_size=20, learning_rate=1e-3,
                 fraction_signal_to_keep=0.95, cuda=False, verbose=0):
        super().__init__()
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.fraction_signal_to_keep = fraction_signal_to_keep

        self.model_0 = NeuralNetModel(n_steps=n_steps, batch_size=batch_size, learning_rate=learning_rate,
                                      cuda=cuda, verbose=verbose)
        self.filter_0 = ClassifierFilter(self.model_0, fraction_signal_to_keep=self.fraction_signal_to_keep)

        self.model_1 = NeuralNetModel(n_steps=n_steps, batch_size=batch_size, learning_rate=learning_rate,
                                      cuda=cuda, verbose=verbose)

    def fit(self, X, y, sample_weight=None):
        self.model_0.fit(X, y, sample_weight=sample_weight)
        # TODO : select on training or validation ?
        # -> training for now, but validation feels more accurate.
        self.filter_0.fit(X, y, sample_weight)
        X_, y_, sample_weight_ = self.filter_0.filter(X, y, sample_weight)

        self.model_1.fit(X_, y_, sample_weight=sample_weight_)

        return self

    def predict(self, X):
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        return y_pred

    def predict_proba(self, X):
        score = np.zeros(X.shape[0])
        idx = self.filter_0.filter_idx(X)
        proba = self.model_1.predict_proba(X.iloc[idx].copy())
        score[idx] = proba[:, 1]
        score = score.reshape(-1, 1)
        proba = np.concatenate([1 - score, score], axis=1)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'model_0')
        if not os.path.isdir(path):
            os.mkdir(path)
        self.model_0.save(path)

        path = os.path.join(dir_path, 'filter_0.json')
        self.filter_0.save_state(path)

        path = os.path.join(dir_path, 'model_1')
        if not os.path.isdir(path):
            os.mkdir(path)
        self.model_1.save(path)
        return self

    def load(self, dir_path):
        path = os.path.join(dir_path, 'model_0')
        self.model_0.load(path)

        path = os.path.join(dir_path, 'filter_0.json')
        self.filter_0.load_state(path)

        path = os.path.join(dir_path, 'model_1')
        self.model_1.load(path)
        return self

    def describe(self):
        return dict(name='cascade_neural_net', learning_rate=self.learning_rate,
                    n_steps=self.n_steps, batch_size=self.batch_size, fraction_signal_to_keep=self.fraction_signal_to_keep)

    def get_name(self):
        name = "CascadeNeuralNetModel-{}-{}-{}-{}".format(self.n_steps, self.batch_size, self.learning_rate, self.fraction_signal_to_keep)
        return name
