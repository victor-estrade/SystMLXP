# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import torch
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from ..net.neural_net import NeuralNetClassifier
from ..net.weighted_criterion import WeightedCrossEntropyLoss

from .architecture import Net
from ..data_augment import NormalDataAugmenter

class NeuralNetModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_steps=5000, batch_size=128, learning_rate=1e-3, cuda=False, verbose=0):
        super().__init__()
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        
        self.net = Net()
        
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = WeightedCrossEntropyLoss()
        
        self.scaler = StandardScaler()
        self.clf = NeuralNetClassifier(self.net, self.criterion, self.optimizer, 
                                       n_steps=self.n_steps, batch_size=self.batch_size, cuda=cuda)

    def fit(self, X, y, sample_weight=None):
        X = X.reshape(-1, 28*28) / 255
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        X = X.reshape(-1, 28*28) / 255
        X = self.scaler.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        X = X.reshape(-1, 28*28) / 255
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        torch.save(self.net.state_dict(), path)
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)
        return self
    
    def load(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        if self.cuda:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self
    
    def describe(self):
        return dict(name='neural_net', learning_rate=self.learning_rate, 
                    n_steps=self.n_steps, batch_size=self.batch_size)
        
    def get_name(self):
        name = "NeuralNetModel-{}-{}-{}".format(self.n_steps, self.batch_size, self.learning_rate)
        return name


class AugmentedNeuralNetModel(BaseEstimator, ClassifierMixin):
    def __init__(self, skewing_function, n_steps=5000, batch_size=128, learning_rate=1e-3, width=1, n_augment=2,
                 cuda=False, verbose=0):
        super().__init__()
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.width = width
        self.n_augment = n_augment
        self.cuda = cuda
        self.verbose = verbose
        
        self.net = Net()
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = WeightedCrossEntropyLoss()
        
        self.augmenter = NormalDataAugmenter(skewing_function, width=width, n_augment=n_augment)

        self.scaler = StandardScaler()
        self.clf = NeuralNetClassifier(self.net, self.criterion, self.optimizer, 
                                       n_steps=self.n_steps, batch_size=self.batch_size, cuda=cuda)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight, z = self.augmenter(X, y, sample_weight)
        X = X.reshape(-1, 28*28)  / 255
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        X = X.reshape(-1, 28*28)  / 255
        X = self.scaler.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        X = X.reshape(-1, 28*28)  / 255
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        torch.save(self.net.state_dict(), path)
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)
        return self
    
    def load(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        if self.cuda:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self

    def describe(self):
        return dict(name='augmented_neural_net', learning_rate=self.learning_rate, 
                    n_steps=self.n_steps, batch_size=self.batch_size, width=self.width, n_augment=self.n_augment)
        
    def get_name(self):
        name = "AugmentedNeuralNetModel-{}-{}-{}-{}-{}".format(self.n_steps, self.batch_size, self.learning_rate,
                        self.width, self.n_augment)
        return name

