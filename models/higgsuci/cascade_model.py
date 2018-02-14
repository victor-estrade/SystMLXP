# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import joblib

from .neural_network_model import NeuralNetModel


def filter_arrays(idx, *arrays):
    filtered_arrays = tuple(arr[idx] if arr is not None else None for arr in arrays)
    return filtered_arrays

class Filter(object):
    def __init__(self, clf, fraction_signal_to_keep=0.95):
        super().__init__()
        self.clf = clf
        self.fraction_signal_to_keep =  fraction_signal_to_keep
        self.score_threshold = 0

    def fit(self, X, y):
        proba = self.clf.predict_proba(X)
        clf_score = proba[:, 1]
        idx = np.argsort(clf_score)
        fraction_signals_kept = np.cumsum(y[idx]) / np.sum(y)
        i = np.searchsorted(fraction_signals_kept, 1-self.fraction_signal_to_keep)
        self.score_threshold = clf_score[idx[i]]
        return self

    def filter_idx(self, X):
        proba = self.clf.predict_proba(X)
        clf_score = proba[:, 1]
        idx = np.argsort(clf_score)
        i = np.searchsorted(clf_score[idx], self.score_threshold)
        return idx[i:]

    def filter(self, X, *arrays):
        idx =  self.filter_idx( X )
        return filter_arrays( idx, X, *arrays )

    def __call__(self, X, *arrays):
        return self.filter(X, *arrays)


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
        self.filter_0 = Filter(self.model_0, fraction_signal_to_keep=self.fraction_signal_to_keep)

        self.model_1 = NeuralNetModel(n_steps=n_steps, batch_size=batch_size, learning_rate=learning_rate,
                                      cuda=cuda, verbose=verbose)

    def fit(self, X, y, sample_weight=None):
        self.model_0.fit(X, y, sample_weight=sample_weight)
        # Select samples keeping 95% signals
        # TODO : select on training or validation ?
        # -> training for now, but validation feels more accurate.
        # TODO Have another class doing the filtering feels cleaner
        self.filter_0.fit(X, y)
        X_, y_, sample_weight_ = self.filter_0.filter(X, y, sample_weight)

        self.model_1.fit(X_, y_, sample_weight=sample_weight_)

        return self
    
    def predict(self, X):
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        return y_pred
    
    def predict_proba(self, X):
        score = np.zeros(X.shape[0])
        idx = self.filter_0.filter_idx(X)
        proba = self.model_1.predict_proba(X[idx])
        score[idx] = proba[:, 1]
        score = score.reshape(-1, 1)
        proba = np.concatenate([1-score, score], axis=1)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'weights_0.pth')
        torch.save(self.net_0.state_dict(), path)
        path = os.path.join(dir_path, 'weights_1.pth')
        torch.save(self.net_1.state_dict(), path)
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)

        path = os.path.join(dir_path, 'losses_0.json')
        self.loss_hook_0.save_state(path)
        path = os.path.join(dir_path, 'losses_1.json')
        self.loss_hook_1.save_state(path)
        return self
    
    def load(self, dir_path):
        path = os.path.join(dir_path, 'weights_0.pth')
        if self.cuda:
            self.net_0.load_state_dict(torch.load(path))
        else:
            self.net_0.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'weights_1.pth')
        if self.cuda:
            self.net_1.load_state_dict(torch.load(path))
        else:
            self.net_1.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)

        path = os.path.join(dir_path, 'losses_0.json')
        self.loss_hook_0.load_state(path)
        path = os.path.join(dir_path, 'losses_1.json')
        self.loss_hook_1.load_state(path)
        return self
    
    def describe(self):
        return dict(name='cascade_neural_net', learning_rate=self.learning_rate, 
                    n_steps=self.n_steps, batch_size=self.batch_size)
        
    def get_name(self):
        name = "CascadeNeuralNetModel-{}-{}-{}".format(self.n_steps, self.batch_size, self.learning_rate)
        return name

