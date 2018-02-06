# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from ..net.tangent_prop import Bias
from ..net.tangent_prop import DSoftPlus


class Net(nn.Module):
    def __init__(self, n_in=14, n_out=2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()


class JNet(nn.Module):
    def __init__(self, n_in=14, n_out=2):
        super(JNet, self).__init__()
        self.fc1 = nn.Linear(n_in, 120, bias=False)
        self.bias1 = Bias(n_in, 120)
        self.fc2 = nn.Linear(120, 120, bias=False)
        self.bias2 = Bias(120, 120)
        self.fc3 = nn.Linear(120, 120, bias=False)
        self.bias3 = Bias(120, 120)
        self.fc4 = nn.Linear(120, n_out, bias=False)
        self.bias4 = Bias(120, n_out)

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


class RNet(nn.Module):
    def __init__(self, n_in=2, n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()