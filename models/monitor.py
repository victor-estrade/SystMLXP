# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import json


class LightLossMonitorHook(object):
    def __init__(self):
        super().__init__()
        self.losses = []

    def __call__(self, module, input, output):
        self.losses.append(output.item())

    def reset(self):
        self.losses = []

    def save_state(self, path):
        with open(path, 'w') as f:
            json.dump(self.losses, f)

    def load_state(self, path):
        with open(path, 'r') as f:
            self.losses = json.load(f)


class LossMonitorHook(object):
    def __init__(self, step=1):
        super().__init__()
        self.losses = []
        self.step = step
        self.i = 0

    def __call__(self, module, input, output):
        self.i += 1
        if self.i >= self.step:
            self.losses.append(output.item())
            self.i = 0

    def reset(self):
        self.losses = []
        self.i = 0

    def save_state(self, path):
        with open(path, 'w') as f:
            data = dict(losses=self.losses,
                        step=self.step,
                        i=self.i,
                        )
            json.dump(data, f)

    def load_state(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.losses = data['losses']
            self.step = data['step']
            self.i = data['i']
