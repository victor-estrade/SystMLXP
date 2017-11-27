# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

#=====================================================================
# Define some model
#=====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.neural_net import NeuralNet

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
N_STEPS = 10000
WIDTH = 15
VERBOSE = 0

PARAMS = dict(learning_rate=LEARNING_RATE, verbose=0, save_step=100)


class Net(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, n_classes)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()


def get_model( n_features=29, n_classes=2, learning_rate=LEARNING_RATE, verbose=VERBOSE, 
              batch_size=BATCH_SIZE, n_steps=N_STEPS, width=WIDTH, save_step=100, cuda=True,
              preprocessing=None, skew=None ):
    def data_augment(X, y, W, training=True):
        if training:
            z = np.random.normal( loc=0, scale=WIDTH, size=(X.shape[0]) )
            X = skew(X, z)
        if preprocessing is not None:
            X, y, W = preprocessing(X, y, W)
        return X, y, W

    net = Net(n_features, n_classes)
    model = NeuralNet(net, n_classes=n_classes, learning_rate=learning_rate, preprocessing=data_augment, verbose=verbose,
                      batch_size=batch_size, n_steps=n_steps, width=width, save_step=save_step, cuda=cuda)
    model.name = 'DataAugmentNeuralNet'
    return model
