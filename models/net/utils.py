#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import Variable

def make_variable(arr, cuda=True, volatile=False):
    """
    Transform given numpy array to a torch.autograd.Variable
    """
    var = Variable(torch.from_numpy(arr), volatile=volatile)
    if cuda:
        var = var.cuda()
    return var
