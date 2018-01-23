#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from itertools import islice

from .minibatch import OneEpoch, EpochShuffle, OneEpochShuffle
from .weighted_criterion import WeightedCrossEntropyLoss
from .weighted_criterion import to_one_hot

__doc__="""
TODO : Add code
"""
__version__ = "0.1"
__author__ = "Victor Estrade"



def make_variable(arr, cuda=True, volatile=False):
    var = Variable(torch.from_numpy(arr), volatile=volatile)
    if cuda:
        var = var.cuda()
    return var


class TangentPropagation(BaseEstimator, ClassifierMixin):
    def __init__(self, net, tan_function, n_classes=2, learning_rate=1e-3, batch_size=128, trade_off=0.1,
                n_steps=1000, preprocessing=None, verbose=0, save_step=100, cuda=True):
        super().__init__()
        self.name = 'TangentPropagation'
        self.tan_function = tan_function
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.trade_off = trade_off
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.save_step = save_step
        self.cuda = cuda
        if cuda:
            self.net = net.cuda()
        else:
            self.net = net
        self.scaler = StandardScaler()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = WeightedCrossEntropyLoss().cuda()

    def get_name(self):
        name = "{}-{}-{}-{}".format(self.name, self.learning_rate, self.trade_off, self.n_steps)
        return name

    def save(self, dir_path):
        """Save the model in the given directory"""
        path = os.path.join(dir_path, 'weights.pth')
        torch.save(self.net.state_dict(), path)
        
        path = os.path.join(dir_path, 'losses.csv')
        losses = np.array(self.losses)
        np.savetxt(path, losses)

        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)
        return self
    
    def load(self, dir_path):
        """Load the model of th i-th CV from the given directory"""
        path = os.path.join(dir_path, 'weights.pth')
        if self.cuda:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        
        path = os.path.join(dir_path, 'losses.csv')
        losses = np.loadtxt(path)
        self.losses = losses
        
        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self
    
    def fit(self, X, y, sample_weight=None, batch_size=256, n_steps=None, n_epochs=None):
        if n_steps is None and n_epochs is None:
            n_steps = self.n_steps
        if n_steps is not None and n_epochs is not None:
            raise ValueError('n_steps and n_epochs cannot be both specified.')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        T = self.tan_function(X)

        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        if isinstance(y, pd.core.generic.NDFrame):
            y = y.values
        if isinstance(sample_weight, pd.core.generic.NDFrame):
            sample_weight = sample_weight.values
        if isinstance(T, pd.core.generic.NDFrame):
            T = T.values

        self.net.reset_parameters()
        self.losses = []
        
        if self.preprocessing is not None:
            X, y, sample_weight = self.preprocessing(X, y, sample_weight, training=True)
        X = self.scaler.fit_transform(X)
        X = X.astype(np.float32)
        T = T.astype(np.float32).reshape(*X.shape)  # in case preprocessing does some reshape
        sample_weight = sample_weight.astype(np.float32)
        y = y.astype(np.int64)
        
        batch_size = self.batch_size
        if n_steps is not None:
            self._fit_steps(X, y, T, sample_weight, batch_size, n_steps)
        if n_epochs is not None:
            self._fit_epochs(X, y, T, sample_weight, batch_size, n_epochs)
        
        return self
        
    def _fit_epochs(self, X, y, T, sample_weight, batch_size, n_epochs):
        for epoch in range(n_epochs):
            batch_gen = OneEpochShuffle(X, y, T, sample_weight, batch_size=batch_size)
            batch_gen.epoch = epoch
            for i, (X_batch, y_batch, T_batch, w_batch) in enumerate(batch_gen):
                self._fit_batch(X_batch, y_batch, T_batch, w_batch, batch_gen)
                
    def _fit_steps(self, X, y, T, sample_weight, batch_size, n_steps):
        batch_gen = EpochShuffle(X, y, T, sample_weight, batch_size=batch_size)
        for i, (X_batch, y_batch, T_batch, w_batch) in enumerate(islice(batch_gen, n_steps)):
            self._fit_batch(X_batch, y_batch, T_batch, w_batch, batch_gen)
            
    def _fit_batch(self, X_batch, y_batch, T_batch, w_batch, batch_gen):
        self.net.train() # train mode
        X_batch = make_variable(X_batch, cuda=self.cuda)
        T_batch = make_variable(T_batch, cuda=self.cuda)
        w_batch = make_variable(w_batch, cuda=self.cuda)
        y_batch = make_variable(to_one_hot(y_batch, n_class=self.n_classes), cuda=self.cuda)
        self.optimizer.zero_grad() # zero-out the gradients because they accumulate by default
        
        y_pred, j_pred = self.net(X_batch, T_batch)
        loss = self.criterion(y_pred, y_batch, w_batch) 
        jloss = torch.sum(j_pred ** 2, 1) * w_batch
        loss = loss + self.trade_off *  torch.mean(jloss)
        
        loss.backward() # compute gradients
        self.optimizer.step() # update params
        if (batch_gen.step % self.save_step) == 0 :
            self.losses.append(loss.data[0])
            if self.verbose:
                print( 'Epoch {}, [{} / {} samples] : loss = {}'.format(batch_gen.epoch, batch_gen.yielded, batch_gen.size, loss.data[0]) )

    def predict_proba(self, X, batch_size=1024):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        if self.preprocessing is not None:
            X, _, _ = self.preprocessing(X, None, None, training=False)
        X = self.scaler.transform(X)
        batch_gen = OneEpoch(X, batch_size=batch_size)
        y_proba = []
        self.net.eval()
        for X_batch in batch_gen:
            X_batch = X_batch.astype(np.float32)
            X_batch = make_variable(X_batch, cuda=self.cuda, volatile=True)
            out, _ = self.net(X_batch, X_batch)
            proba_batch = nn.Softmax(dim=1)(out).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba 

    def predict(self, X, batch_size=1024):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        return np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)
    
    def score(self, X, y, batch_size=1024):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        if isinstance(y, pd.core.generic.NDFrame):
            y = y.values
        pred = np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)
        return np.mean( pred == y )


class Bias(nn.Module):
    r"""Applies a add transformation to the incoming data: :math:`y = x + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Attributes:
        bias:   the learnable bias of the module of shape (out_features)
    """

    def __init__(self, in_features, out_features):
        super(Bias, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input.add(self.bias.expand_as(input))
        return input

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.out_features) + ' -> ' \
            + str(self.out_features) + ')'


class DSigmoid(nn.Module):
    """Applies the element-wise function :math:`f(x) = sigmoid(x) * (1 - sigmoid(x))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        x = torch.sigmoid(input)
        return x * ( 1 - x )

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class DTanh(nn.Module):
    """Applies the element-wise function :math:`f(x) = 1 - (tanh(x))^2`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        x = torch.tanh(input)
        return 1 - (x * x)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class DSoftPlus(nn.Module):
    """Applies the element-wise function :math:`f(x) = sigmoid(x)`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def forward(self, input):
        return torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class DSoftSign(nn.Module):
    """Applies the element-wise function :math:`f(x) = sigmoid(x)`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def forward(self, input):
        x = ( 1.0 + torch.abs(input) )
        return 1.0 / ( x * x )

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

