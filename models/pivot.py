#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import islice

from .minibatch import OneEpoch, EpochShuffle, OneEpochShuffle
from .weighted_criterion import WeightedCrossEntropyLoss
from .weighted_criterion import to_one_hot

__doc__="""
TODO : Add code
"""
__version__ = "0.1"
__author__ = "Victor Estrade"

from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import islice

from models.minibatch import OneEpoch, EpochShuffle, OneEpochShuffle
from models.weighted_criterion import WeightedBinaryCrossEntropyLoss
from models.weighted_criterion import WeightedMSELoss

def make_variable(arr, cuda=True, volatile=False):
    var = Variable(torch.from_numpy(arr), volatile=volatile)
    if cuda:
        var = var.cuda()
    return var

class Pivot(BaseEstimator, ClassifierMixin):
    def __init__(self, dnet, rnet, skew, lr=1e-3, batch_size=128, trade_of=1.0, verbose=0, save_step=100, cuda=True):
        super().__init__()
        self.name = 'Pivot'
        self.scaler = StandardScaler()
        self.lr = lr
        self.batch_size = batch_size
        self.trade_of = trade_of
        self.verbose = verbose
        self.save_step = save_step
        self.cuda = cuda
        if cuda:
            self.dnet = dnet.cuda()
            self.rnet = rnet.cuda()
        else:
            self.dnet = dnet
            self.rnet = rnet
        self.skew = skew
        self.optimizer_D = optim.Adam(self.dnet.parameters(), lr=self.lr)
        self.optimizer_R = optim.SGD(self.rnet.parameters(), lr=self.lr)
        self.optimizer_DR = optim.SGD(list(self.rnet.parameters())+list(self.dnet.parameters()), lr=self.lr)
        
        self.criterion_D = WeightedCrossEntropyLoss().cuda()
        self.criterion_R = WeightedMSELoss().cuda()
    
    def fit(self, X, y, sample_weight=None, n_steps=None, n_epochs=None):
        
        if n_steps is None and n_epochs is None:
            raise ValueError('should give n_steps OR n_epochs, both are None.')
        if n_steps is not None and n_epochs is not None:
            raise ValueError('n_steps and n_epochs cannot be both specified.')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        z = np.random.normal( loc=1, scale=1e-2, size=(X.shape[0]) )
        X = self.skew(X, z)
        
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        if isinstance(y, pd.core.generic.NDFrame):
            y = y.values
        if isinstance(sample_weight, pd.core.generic.NDFrame):
            sample_weight = sample_weight.values

        self.dnet.reset_parameters()
        self.rnet.reset_parameters()
        self.losses_D = []
        self.losses_R = []
        self.losses_DR = []
        
        X = X.astype(np.float32)
        sample_weight = sample_weight.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        
        X = self.scaler.fit_transform(X)
        
        # Pretraining
        if self.verbose:
            print('Pretraining D')
        self._fit_Depochs(X, y, sample_weight, batch_size=batch_size, n_epochs=1)
        
        if self.verbose:
            print('Pretraining R')
        self._fit_Repochs(X, z, sample_weight, batch_size=batch_size, n_epochs=1)
        
        # Adjusting
        N_STEPS = 2000
        batch_size = self.batch_size
        if self.verbose:
            print('Training for', N_STEPS, 'steps')
        batch_gen_DR = EpochShuffle(X, y, z, sample_weight, batch_size=batch_size)
        batch_gen_R = EpochShuffle(X, z, sample_weight, batch_size=batch_size)
        for i, (X_batch, y_batch, z_batch, w_batch) in enumerate(islice(batch_gen_DR, N_STEPS)):
            self._fit_DRbatch(X_batch, y_batch, z_batch, w_batch, batch_gen_DR)
            for i, (X_batch, z_batch, w_batch) in enumerate(islice(batch_gen_R, 20)):
                self._fit_Rbatch(X_batch, z_batch, w_batch, batch_gen_R)

        return self
        
    def _fit_Depochs(self, X, y, sample_weight, batch_size, n_epochs):
        for epoch in range(n_epochs):
            batch_gen = OneEpochShuffle(X, y, sample_weight, batch_size=batch_size)
            batch_gen.epoch = epoch
            for i, (X_batch, y_batch, w_batch) in enumerate(batch_gen):
                self._fit_Dbatch(X_batch, y_batch, w_batch, batch_gen)

    def _fit_Repochs(self, X, z, sample_weight, batch_size, n_epochs):
        for epoch in range(n_epochs):
            batch_gen = OneEpochShuffle(X, z, sample_weight, batch_size=batch_size)
            batch_gen.epoch = epoch
            for i, (X_batch, z_batch, w_batch) in enumerate(batch_gen):
                self._fit_Rbatch(X_batch, z_batch, w_batch, batch_gen)

    def _fit_Dsteps(self, X, y, sample_weight, batch_size, n_steps):
        batch_gen = EpochShuffle(X, y, sample_weight, batch_size=batch_size)

        for i, (X_batch, y_batch, w_batch) in enumerate(islice(batch_gen, n_steps)):
            self._fit_Dbatch(X_batch, y_batch, w_batch, batch_gen)

    def _fit_Rsteps(self, X, z, sample_weight, batch_size, n_steps):
        batch_gen = EpochShuffle(X, z, sample_weight, batch_size=batch_size)

        for i, (X_batch, z_batch, w_batch) in enumerate(islice(batch_gen, n_steps)):
            self._fit_Rbatch(X_batch, z_batch, w_batch, batch_gen)

    def _fit_Dbatch(self, X_batch, y_batch, w_batch, batch_gen):
        self.dnet.train() # train mode
        X_batch = make_variable(X_batch, cuda=self.cuda)
        y_batch = make_variable(y_batch, cuda=self.cuda)
        w_batch = make_variable(w_batch, cuda=self.cuda)
        self.optimizer_D.zero_grad() # zero-out the gradients because they accumulate by default
        y_pred = self.dnet(X_batch)
        loss = self.criterion_D(y_pred, y_batch, w_batch)
        loss.backward() # compute gradients
        self.optimizer_D.step() # update params
        if (batch_gen.step % self.save_step) == 0 :
            self.losses_D.append(loss.data[0])
            if self.verbose:
                print( 'D--Epoch {}, [{} / {} samples] : loss = {}'.format(batch_gen.epoch, batch_gen.yielded % batch_gen.size, 
                                                                       batch_gen.size, loss.data[0]) )

    def _fit_Rbatch(self, X_batch, z_batch, w_batch, batch_gen):
        self.dnet.eval()
        self.rnet.train() # train mode
        X_batch = make_variable(X_batch, cuda=self.cuda)
        y_batch = make_variable(y_batch, cuda=self.cuda)
        w_batch = make_variable(w_batch, cuda=self.cuda)
        z_batch = make_variable(z_batch, cuda=self.cuda)
        self.optimizer_R.zero_grad() # zero-out the gradients because they accumulate by default
        y_pred = self.dnet(X_batch)
        z_pred = self.rnet(y_pred)
        loss = self.criterion_R(z_pred.view(-1), z_batch, w_batch)
        loss.backward() # compute gradients
        self.optimizer_R.step() # update params
        if (batch_gen.step % self.save_step) == 0 :
            self.losses_R.append(loss.data[0])
            if self.verbose:
                print( 'R--Epoch {}, [{} / {} samples] : loss = {}'.format(batch_gen.epoch, batch_gen.yielded % batch_gen.size,
                                                                       batch_gen.size, loss.data[0]) )

    def _fit_DRbatch(self, X_batch, y_batch, z_batch, w_batch, batch_gen):
        self.dnet.train()
        self.rnet.train() # train mode
        X_batch = make_variable(X_batch, cuda=self.cuda)
        y_batch = make_variable(y_batch, cuda=self.cuda)
        w_batch = make_variable(w_batch, cuda=self.cuda)
        z_batch = make_variable(z_batch, cuda=self.cuda)
        self.optimizer_DR.zero_grad() # zero-out the gradients because they accumulate by default
        y_pred = self.dnet(X_batch)
        z_pred = self.rnet(y_pred)
        loss_D = self.criterion_D(y_pred, y_batch, w_batch)
        loss_R = self.criterion_R(z_pred.view(-1), z_batch, w_batch)
        loss = loss_D - ( self.trade_of * loss_R )
        loss.backward() # compute gradients
        self.optimizer_DR.step() # update params
        if (batch_gen.step % self.save_step) == 0 :
            self.losses_DR.append(loss.data[0])
            if self.verbose:
                print( 'DR--Epoch {}, [{} / {} samples] : loss = {}'.format(batch_gen.epoch, batch_gen.yielded % batch_gen.size, 
                                                                            batch_gen.size, loss.data[0]) )
                print( '                                 Dloss = {}'.format(loss_D.data[0]) )
                print( '                                 Rloss = {}'.format(loss_R.data[0]) )

    def predict_proba(self, X, batch_size=200):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        X = self.scaler.transform(X)
        batch_gen = OneEpoch(X, batch_size=batch_size)
        y_proba = []
        self.dnet.eval()
        for X_batch in batch_gen:
            X_batch = X_batch.astype(np.float32)
            X_batch = make_variable(X_batch, volatile=True, cuda=self.cuda)
            proba_batch = F.softmax(self.dnet(X_batch)).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        if y_proba.shape[1] == 1:
            y_proba = np.concatenate([1-y_proba, y_proba], axis=1)
        return y_proba 

    def predict(self, X, batch_size=200):
        proba = self.predict_proba(X, batch_size=batch_size)
        y_pred = np.argmax(proba, axis=1)
        return y_pred.astype(np.int64)
    
    def score(self, X, y, batch_size=200):
        if isinstance(y, pd.core.generic.NDFrame):
            y = y.values
        pred = self.predict(X, batch_size=batch_size)
        return np.mean( pred == y )
