# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import numpy as np

import torch
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from ..net.tangent_prop import TangentPropClassifier
from ..net.weighted_criterion import WeightedCrossEntropyLoss
from ..net.weighted_criterion import WeightedL2Loss
from ..tangent_extract import TangentExtractor

from .architecture import JNet
from ..data_augment import NormalDataAugmenter

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


class TangentPropModel(BaseEstimator, ClassifierMixin):
    def __init__(self, skewing_function, n_steps=5000, batch_size=20, learning_rate=1e-3, trade_off=1, alpha=1e-2, cuda=False, verbose=0):
        super().__init__()
        self.skewing_function = skewing_function
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trade_off = trade_off
        self.alpha = alpha
        self.cuda = cuda
        self.verbose = verbose
        
        self.jnet = JNet()
        
        self.optimizer = optim.Adam(self.jnet.parameters(), lr=learning_rate)
        self.criterion = WeightedCrossEntropyLoss()
        self.jcriterion = WeightedL2Loss()
        
        self.tangent_extractor = TangentExtractor(skewing_function, alpha=alpha)
#         self.tangent_extractor = TangentComputer()

        self.scaler = StandardScaler()
        self.clf = TangentPropClassifier(self.jnet, self.criterion, self.jcriterion, self.optimizer, 
                                         n_steps=self.n_steps, batch_size=self.batch_size,
                                         trade_off=trade_off, cuda=cuda)

    def fit(self, X, y, sample_weight=None):
        T = self.tangent_extractor.compute_tangent(X)
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y, T, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        torch.save(self.jnet.state_dict(), path)
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)
        return self
    
    def load(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        if self.cuda:
            self.jnet.load_state_dict(torch.load(path))
        else:
            self.jnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self
    
    def describe(self):
        return dict(name='tangent_prop', n_steps=self.n_steps, batch_size=self.batch_size, 
                    learning_rate=self.learning_rate, trade_off=self.trade_off, alpha=self.alpha)
    
    def get_name(self):
        name = "TangentPropModel-{}-{}-{}-{}-{}".format(self.n_steps, self.batch_size, 
                            self.learning_rate, self.trade_off, self.alpha)
        return name


class AugmentedTangentPropModel(BaseEstimator, ClassifierMixin):
    def __init__(self, skewing_function, n_steps=5000, batch_size=20, learning_rate=1e-3, trade_off=1, alpha=1e-2, 
                    width=1, n_augment=2, cuda=False, verbose=0):
        super().__init__()
        self.skewing_function = skewing_function
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trade_off = trade_off
        self.alpha = alpha
        self.width = width
        self.n_augment = n_augment
        self.cuda = cuda
        self.verbose = verbose
        
        self.jnet = JNet()
        
        self.optimizer = optim.Adam(self.jnet.parameters(), lr=learning_rate)
        self.criterion = WeightedCrossEntropyLoss()
        
        self.tangent_extractor = TangentExtractor(skewing_function, alpha=alpha)
#         self.tangent_extractor = TangentComputer()

        self.augmenter = NormalDataAugmenter(skewing_function, width=width, n_augment=n_augment)

        self.scaler = StandardScaler()
        self.clf = TangentPropClassifier(self.jnet, self.criterion, self.optimizer, 
                                         n_steps=self.n_steps, batch_size=self.batch_size,
                                         trade_off=trade_off, cuda=cuda)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight, z = self.augmenter(X, y, sample_weight)
        T = self.tangent_extractor.compute_tangent(X)
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y, T, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        torch.save(self.jnet.state_dict(), path)
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)
        return self
    
    def load(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        if self.cuda:
            self.jnet.load_state_dict(torch.load(path))
        else:
            self.jnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self
    
    def describe(self):
        return dict(name='tangent_prop', n_steps=self.n_steps, batch_size=self.batch_size, 
                    learning_rate=self.learning_rate, trade_off=self.trade_off, alpha=self.alpha)
    
    def get_name(self):
        name = "TangentPropModel-{}-{}-{}-{}-{}".format(self.n_steps, self.batch_size, 
                            self.learning_rate, self.trade_off, self.alpha)
        return name

