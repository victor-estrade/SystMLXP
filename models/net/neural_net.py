#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

from itertools import islice
from .minibatch import EpochShuffle
from .minibatch import OneEpoch
from .utils import make_variable

__version__ = "0.1"
__author__ = "Victor Estrade"


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, net, criterion, optimizer, n_steps, batch_size, cuda=False, verbose=0):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.cuda = cuda
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.criterion = self.criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.criterion = self.criterion.cpu()

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        X = X.astype(np.float32)
        sample_weight = sample_weight.astype(np.float32)
        y = y.astype(np.int64)
        
        batch_size = self.batch_size
        n_steps = self.n_steps
        
        self.net.reset_parameters()
        
        batch_gen = EpochShuffle(X, y, sample_weight, batch_size=batch_size)
        for i, (X_batch, y_batch, w_batch) in enumerate(islice(batch_gen, n_steps)):
            X_batch = make_variable(X_batch, cuda=self.cuda)
            w_batch = make_variable(w_batch, cuda=self.cuda)
            y_batch = make_variable(y_batch, cuda=self.cuda)
            self.net.train() # train mode
            self.optimizer.zero_grad() # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            loss = self.criterion(y_pred, y_batch, w_batch)
            loss.backward() # compute gradients
            self.optimizer.step() # update params
            # TODO : Call epoch hook. Compute i*batch_size/epoch to catch epoch's end
            # RMQ : Or maybe hooks should be handle by torch.Module or my TrainableNet ?
        return self
    
    def predict(self, X, batch_size=None):
        return np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)
    
    def predict_proba(self, X, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch_gen = OneEpoch(X, batch_size=batch_size)
        y_proba = []
        self.net.eval()
        for X_batch in batch_gen:
            X_batch = X_batch.astype(np.float32)
            X_batch = make_variable(X_batch, cuda=self.cuda, volatile=True)
            proba_batch = F.softmax(self.net.forward(X_batch), dim=1).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba 

class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, net, criterion, optimizer, n_steps, batch_size, cuda=False, verbose=0):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.cuda = cuda
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.criterion = self.criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.criterion = self.criterion.cpu()

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        X = X.astype(np.float32)
        sample_weight = sample_weight.astype(np.float32)
        y = y.astype(np.float32)
        
        batch_size = self.batch_size
        n_steps = self.n_steps
        
        self.net.reset_parameters()
        
        batch_gen = EpochShuffle(X, y, sample_weight, batch_size=batch_size)
        self.net.train() # train mode
        for i, (X_batch, y_batch, w_batch) in enumerate(islice(batch_gen, n_steps)):
            X_batch = make_variable(X_batch, cuda=self.cuda)
            w_batch = make_variable(w_batch, cuda=self.cuda)
            y_batch = make_variable(y_batch, cuda=self.cuda)
            self.optimizer.zero_grad() # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            loss = self.criterion(y_pred, y_batch, w_batch)
            loss.backward() # compute gradients
            self.optimizer.step() # update params
            # TODO : Call epoch hook. Compute i*batch_size/epoch to catch epoch's end
            # RMQ : Or maybe hooks should be handle by torch.Module or my TrainableNet ?
        return self
    
    def predict(self, X, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch_gen = OneEpoch(X, batch_size=batch_size)
        y_pred = []
        self.net.eval()
        for X_batch in batch_gen:
            X_batch = X_batch.astype(np.float32)
            X_batch = make_variable(X_batch, cuda=self.cuda, volatile=True)
            pred_batch = self.net.forward(X_batch).cpu().data.numpy()
            y_pred.extend(pred_batch)
        y_pred = np.array(y_pred)
        return y_pred 
