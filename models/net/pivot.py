# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from itertools import islice
from .minibatch import EpochShuffle
from .utils import make_variable

__version__ = "0.1"
__author__ = "Victor Estrade"

class PivotTrainer(object):
    def __init__(self, classifier, adversarial, droptimizer, n_steps=1000, batch_size=20, n_recovery_steps=20,
                 trade_off=1, cuda=False, verbose=0):
        super().__init__()
        self.classifier = classifier
        self.adversarial = adversarial
        self.droptimizer = droptimizer
        self.n_steps = n_steps
        self.n_recovery_steps = n_recovery_steps
        self.batch_size = batch_size
        self.trade_off = trade_off
        self.cuda = cuda

    def partial_fit(self, X, y, z, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)        
        
        X = X.astype(np.float32)
        z = z.astype(np.float32)
        sample_weight = sample_weight.astype(np.float32)
        y = y.astype(np.int64)

        batch_size = self.batch_size
        
        batch_gen_DR = EpochShuffle(X, y, z, sample_weight, batch_size=batch_size)
        batch_gen_R = EpochShuffle(X, z, sample_weight, batch_size=batch_size)
        self.classifier.net.train() # train mode
        self.adversarial.net.train() # train mode
        for i, (X_batch, y_batch, z_batch, w_batch) in enumerate(islice(batch_gen_DR, self.n_steps)):
            X_batch = make_variable(X_batch, cuda=self.cuda)
            z_batch = make_variable(z_batch, cuda=self.cuda)
            w_batch = make_variable(w_batch, cuda=self.cuda)
            y_batch = make_variable(y_batch, cuda=self.cuda)
            self.droptimizer.zero_grad() # zero-out the gradients because they accumulate by default
            y_pred = self.classifier.net(X_batch)
            z_pred = self.adversarial.net(y_pred)
            loss_clf = self.classifier.criterion(y_pred, y_batch, w_batch) 
            loss_adv = self.adversarial.criterion(z_pred, z_batch, w_batch)
            loss = loss_clf - ( self.trade_off * loss_adv )
            loss.backward() # compute gradients
            self.droptimizer.step() # update params

            for j, (X_batch, z_batch, w_batch) in enumerate(islice(batch_gen_R, self.n_recovery_steps)):
                X_batch = make_variable(X_batch, cuda=self.cuda)
                z_batch = make_variable(z_batch, cuda=self.cuda)
                w_batch = make_variable(w_batch, cuda=self.cuda)
                self.adversarial.optimizer.zero_grad() # zero-out the gradients because they accumulate by default
                y_pred = self.classifier.net(X_batch)
                z_pred = self.adversarial.net(y_pred)
                loss = self.adversarial.criterion(z_pred, z_batch, w_batch)
                loss.backward() # compute gradients
                self.adversarial.optimizer.step() # update params
        return self
    
    def predict(self, X):
        raise NotImplementedError
    
    def predict_proba(self, X):
        raise NotImplementedError
