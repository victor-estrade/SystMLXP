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

from ..net.neural_net import NeuralNetClassifier
from ..net.neural_net import NeuralNetRegressor
from ..net.pivot import PivotTrainer
from ..net.weighted_criterion import WeightedCrossEntropyLoss
from ..net.weighted_criterion import WeightedMSELoss

from .architecture import Net
from .architecture import RNet
from ..data_augment import NormalDataAugmenter


class ZComputer(object):
    def __init__(self):
        super().__init__()

    def compute_z(self, X):
        x = X[:, 0]
        y = X[:, 1]
        z = np.arctan2(y, x)
        return z


class PivotModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clf_pre_training_steps=10, n_adv_pre_training_steps=10, n_steps=1000, n_recovery_steps=10,
                 batch_size=128, classifier_learning_rate=1e-3, adversarial_learning_rate=1e-3, trade_off=1,
                 cuda=False, verbose=0):
        super().__init__()
        self.n_clf_pre_training_steps = n_clf_pre_training_steps
        self.n_adv_pre_training_steps = n_adv_pre_training_steps
        self.n_steps = n_steps
        self.n_recovery_steps = n_recovery_steps
        self.batch_size = batch_size
        self.classifier_learning_rate = classifier_learning_rate
        self.adversarial_learning_rate = adversarial_learning_rate
        self.trade_off = trade_off
        self.cuda = cuda
        self.verbose = verbose
        
        self.dnet = Net()
        self.rnet = RNet()
        
        self.doptimizer = optim.Adam(self.dnet.parameters(), lr=classifier_learning_rate)
        self.dcriterion = WeightedCrossEntropyLoss()
        self.classifier = NeuralNetClassifier(self.dnet, self.dcriterion, self.doptimizer, 
                                              n_steps=n_clf_pre_training_steps, batch_size=batch_size, cuda=self.cuda)

        self.roptimizer = optim.Adam(self.rnet.parameters(), lr=adversarial_learning_rate)
        self.rcriterion = WeightedMSELoss()
#         self.regressor = NeuralNetRegressor(nn.Sequential(self.dnet, self.rnet), self.dcriterion, self.doptimizer, 
#                                               n_steps=n_adv_pre_training_steps, batch_size=batch_size, cuda=self.cuda)

        self.adversarial = NeuralNetRegressor(self.rnet, self.rcriterion, self.roptimizer, 
                                              n_steps=n_adv_pre_training_steps, batch_size=batch_size, cuda=self.cuda)
        
        self.droptimizer = optim.Adam(list(self.dnet.parameters()) + list(self.rnet.parameters()), lr=adversarial_learning_rate)
        self.pivot = PivotTrainer(self.classifier, self.adversarial, self.droptimizer,
                                 n_steps=self.n_steps, n_recovery_steps=n_recovery_steps, batch_size=batch_size,
                                 trade_off=trade_off)
        
        self.zcomputer = ZComputer()
        self.scaler = StandardScaler()
        
    def fit(self, X, y, sample_weight=None):
        z = self.zcomputer.compute_z(X)
        X = X.reshape(-1, 28*28)
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y, sample_weight=sample_weight)  # pre-training
        y_pred = self.classifier.predict_proba(X)
        self.adversarial.fit(y_pred, z, sample_weight=sample_weight)  # pre-training
        self.pivot.partial_fit(X, y, z, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        X = X.reshape(-1, 28*28)
        X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        X = X.reshape(-1, 28*28)
        X = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'dnet_weights.pth')
        torch.save(self.dnet.state_dict(), path)

        path = os.path.join(dir_path, 'rnet_weights.pth')
        torch.save(self.rnet.state_dict(), path)
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)
        return self
    
    def load(self, dir_path):
        path = os.path.join(dir_path, 'dnet_weights.pth')
        if self.cuda:
            self.dnet.load_state_dict(torch.load(path))
        else:
            self.dnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'rnet_weights.pth')
        if self.cuda:
            self.rnet.load_state_dict(torch.load(path))
        else:
            self.rnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self
    
    def describe(self):
        return dict(name='pivot', n_clf_pre_training_steps=self.n_clf_pre_training_steps,
                    n_adv_pre_training_steps=self.n_adv_pre_training_steps, n_steps=self.n_steps,
                    n_recovery_steps=self.n_recovery_steps, classifier_learning_rate=self.classifier_learning_rate, 
                    batch_size=self.batch_size,
                    adversarial_learning_rate=self.adversarial_learning_rate, trade_off=self.trade_off,
                    )

    def get_name(self):
        name = "PivotModel-{}-{}-{}-{}-{}-{}-{}-{}".format(self.n_clf_pre_training_steps, self.n_adv_pre_training_steps, 
                    self.n_steps, self.n_recovery_steps, self.classifier_learning_rate, self.batch_size,
                    self.adversarial_learning_rate, self.trade_off,
                    )
        return name
