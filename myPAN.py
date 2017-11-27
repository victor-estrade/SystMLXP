# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

#=====================================================================
# Define some model
#=====================================================================
from models.pan_new import PAN_clasifieur_AGNO

import tensorflow as tf

LEARNING_RATE = 1e-3
TRADE_OFF =  1.0
BATCH_SIZE = 128
N_STEPS = 7000
N_STEPS_PRE_TRAINING = 3000
N_STEPS_CATCH_TRAINING = 5
WIDTH = 5


def get_model(n_features=29, n_classes=2, learning_rate=LEARNING_RATE, trade_off=TRADE_OFF, width=WIDTH,
              n_steps=N_STEPS, n_steps_pre_training=N_STEPS_PRE_TRAINING, 
              n_steps_catch_training=N_STEPS_CATCH_TRAINING, batch_size=BATCH_SIZE,
              skew=None, preprocessing=None):
    INPUT_SHAPE = (n_features, )
    
    model = PAN_clasifieur_AGNO(INPUT_SHAPE, n_classes=n_classes, skew=skew, preprocessing=preprocessing, 
                    learning_rate=learning_rate, trade_off=trade_off, width=width,
                    n_steps=n_steps, n_steps_pre_training=n_steps_pre_training,
                    n_steps_catch_training=n_steps_catch_training, batch_size=batch_size,
                    n_layers_D=3, n_neurons_per_layer_D=120,
                    n_layers_R=3, n_neurons_per_layer_R=120,
                    opti_D="Adam", opti_R="SGD", opti_DR="SGD",
                    )
    return model
