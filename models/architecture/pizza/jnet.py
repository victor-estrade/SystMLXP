# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from ...net.tangent_prop import Bias
from ...net.tangent_prop import DSoftPlus

class JNet(nn.Module):
    def __init__(self):
        super(JNet, self).__init__()
        self.fc1 = nn.Linear(2, 5, bias=False)
        self.bias1 = Bias(2, 5)
        self.fc2 = nn.Linear(5, 5, bias=False)
        self.bias2 = Bias(5, 5)
        self.fc3 = nn.Linear(5, 5, bias=False)
        self.bias3 = Bias(5, 5)
        self.fc4 = nn.Linear(5, 2, bias=False)
        self.bias4 = Bias(5, 2)

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

