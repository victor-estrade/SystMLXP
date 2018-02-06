# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from itertools import islice
from .minibatch import EpochShuffle
from .minibatch import OneEpoch
from .utils import make_variable


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


class TangentPropClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, jnet, criterion, optimizer, n_steps, batch_size, trade_off=1, cuda=False, verbose=0):
        super().__init__()
        self.jnet = jnet
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.trade_off = trade_off
        self.verbose = verbose
        self.cuda_flag = cuda
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.jnet = self.jnet.cuda(device=device)
        self.criterion = self.criterion.cuda(device=device)

    def cpu(self):
        self.jnet = self.jnet.cpu()
        self.criterion = self.criterion.cpu()

    def fit(self, X, y, T, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)    

        X = X.astype(np.float32)
        T = T.astype(np.float32)
        sample_weight = sample_weight.astype(np.float32)
        y = y.astype(np.int64)
        
        batch_size = self.batch_size
        n_steps = self.n_steps
        
        self.jnet.reset_parameters()
        batch_gen = EpochShuffle(X, y, sample_weight, T, batch_size=batch_size)
        for i, (X_batch, y_batch, w_batch, T_batch) in enumerate(islice(batch_gen, n_steps)):
            X_batch = make_variable(X_batch, cuda=self.cuda_flag)
            T_batch = make_variable(T_batch, cuda=self.cuda_flag)
            w_batch = make_variable(w_batch, cuda=self.cuda_flag)
            y_batch = make_variable(y_batch, cuda=self.cuda_flag)
            self.jnet.train() # train mode
            self.optimizer.zero_grad() # zero-out the gradients because they accumulate by default
            y_pred, j_pred = self.jnet(X_batch, T_batch)
            loss = self.criterion(y_pred, y_batch, w_batch) 
            jloss = torch.sum( j_pred * j_pred, 1) * w_batch
            loss = loss + self.trade_off *  torch.mean(jloss)
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
        self.jnet.eval()
        for X_batch in batch_gen:
            X_batch = X_batch.astype(np.float32)
            X_batch = make_variable(X_batch, cuda=self.cuda_flag, volatile=True)
            out, _ = self.jnet(X_batch, X_batch)
            proba_batch = F.softmax(out, dim=1).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba 
