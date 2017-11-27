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

from models.tangent_prop import TangentPropagation
from models.tangent_prop import Bias
from models.tangent_prop import DSoftPlus


class JNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super(JNet, self).__init__()
        self.fc1 = nn.Linear(n_features, 120, bias=False)
        self.bias1 = Bias(n_features, 120)
        self.fc2 = nn.Linear(120, 120, bias=False)
        self.bias2 = Bias(120, 120)
        self.fc3 = nn.Linear(120, 120, bias=False)
        self.bias3 = Bias(120, 120)
        self.fc4 = nn.Linear(120, n_classes, bias=False)
        self.bias4 = Bias(120, n_classes)

    def forward(self, x, jx):
        x = self.bias1(self.fc1(x))
        jx = self.fc1(jx) * DSoftPlus()(x)
        x = F.softplus(x)

        x = self.bias2(self.fc2(x))
        jx = self.fc2(jx) * DSoftPlus()(x)
        x = F.softplus(x)

        x = self.bias3(self.fc3(x))
        jx = self.fc3(jx) * DSoftPlus()(x)
        x = F.softplus(x)

        x = self.bias4(self.fc4(x))
        jx = self.fc4(jx)
        return x, jx

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.bias1.reset_parameters()
        self.bias2.reset_parameters()
        self.bias3.reset_parameters()
        self.bias4.reset_parameters()


LEARNING_RATE = 1e-3
TRADE_OFF = 100.0
BATCH_SIZE = 128
N_STEPS = 10000
VERBOSE = 0

PARAMS = dict(LEARNING_RATE = LEARNING_RATE,
                TRADE_OFF = TRADE_OFF,
                BATCH_SIZE = BATCH_SIZE,
                N_STEPS = N_STEPS,
                )

def get_model( n_features=29, n_classes=2, learning_rate=LEARNING_RATE, trade_off=TRADE_OFF, n_steps=N_STEPS,
               batch_size=BATCH_SIZE, verbose=0, save_step=100, cuda=True,
               tangent=None, preprocessing=None):
    net = JNet(n_features, n_classes)
    model = TangentPropagation(net, tangent, n_classes=n_classes, learning_rate=learning_rate, trade_off=trade_off, n_steps=n_steps,
                               batch_size=batch_size, preprocessing=preprocessing, verbose=verbose, save_step=save_step, cuda=cuda)
    return model
