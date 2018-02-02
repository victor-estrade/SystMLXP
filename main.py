#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import datetime
import os
import argparse
import inspect

import numpy as np
import pandas as pd

from problem.workflow import print

from myNN import get_model as get_model_NN
from myNNA import get_model as get_model_NNA
from myNNDA import get_model as get_model_NNDA
from myTP import get_model as get_model_TP
from myPAN import get_model as get_model_PAN


MODELS = {
        'NN': get_model_NN,
        'NNA': get_model_NNA,
        'NNDA': get_model_NNDA,
        'TP': get_model_TP,
        'PAN': get_model_PAN,
         }

DATA = ['mnist', 'fashion-mnist', 'higgs-geant', 'higgs-uci',]


def parse_args():
    # TODO : more descriptive msg.
    parser = argparse.ArgumentParser(
        description="Training launcher"
        )

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
        default=0, help="increase output verbosity")
    
    # DATASET CHOICE
    parser.add_argument('--data', help='chosen dataset',
        type=str, choices=DATA, default='mnist' )
    
    # MODEL CHOICE
    parser.add_argument('model', help='model to train',
        type=str, choices=MODELS.keys() )
    

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
        default=1e-3, type=float)
    
    parser.add_argument('--trade-off', help='trade-off for multi-objective models',
        default=1.0, type=float)
    
    parser.add_argument('-w', '--width', help='width for the data augmentation sampling',
        default=5, type=float)

    parser.add_argument('--batch-size', help='mini-batch size',
        default=128, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
        default=10000, type=int)

    parser.add_argument('--n-steps-pre-training', 
        help='number of update steps for the pre-training',
        default=3000, type=int)

    parser.add_argument('--n-steps-catch-training', 
        help='number of update steps for the catch training of auxiliary models',
        default=5, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
        action='store_false', dest='cuda')

    args = parser.parse_args()
    return args


def extract_model_args(args, get_model):
    sig = inspect.signature(get_model)
    args_dict = vars(args)
    model_args = { k: args_dict[k] for k in sig.parameters.keys() if k in args_dict.keys() }
    return model_args

def get_data_loader(data_name):
    if data_name == 'mnist':
        from problem.mnist import load_data
    elif data_name == 'fashion-mnist':
        from problem.fashion_mnist import load_data
    elif data_name == 'higgs-geant':
        from problem.higgs_geant import load_data
    elif data_name == 'higgs-uci':
        from problem.higgs_uci import load_data
    else:
        raise ValueError('Unrecognise dataset name : {}'
                         'Expected one from {}'. format(data_name, DATA))
    return load_data


def get_data_shape(data_name):
    """ Return n_features, n_classes"""
    if data_name == 'mnist':
        return 28*28, 10
    elif data_name == 'fashion-mnist':
        return 28*28, 10
    elif data_name == 'higgs-geant':
        return 29, 2
    elif data_name == 'higgs-uci':
        return 14, 2
    else:
        raise ValueError('Unrecognise dataset name : {}'
                         'Expected one from {}'. format(data_name, DATA))
        return None


def get_problem_functions(data_name):
    if data_name == 'mnist':
        from problem.mnist import preprocessing
        from problem.mnist import skew
        from problem.mnist import tangent
        from problem.mnist import train_submission

        return train_submission, preprocessing, skew, tangent

    if data_name == 'fashion-mnist':
        from problem.fashion_mnist import preprocessing
        from problem.fashion_mnist import skew
        from problem.fashion_mnist import tangent
        from problem.fashion_mnist import train_submission

        return train_submission, preprocessing, skew, tangent

    elif data_name == 'higgs-geant':
        from problem.higgs_geant import skew
        from problem.higgs_geant import tangent
        from problem.higgs_geant import train_submission

        return train_submission, None, skew, tangent

    elif data_name == 'higgs-uci':
        from problem.higgs_uci import skew
        from problem.higgs_uci import tangent
        from problem.higgs_uci import train_submission

        return train_submission, None, skew, tangent

    else:
        raise ValueError('Unrecognise dataset name : {}'
                         'Expected one from {}'. format(data_name, DATA))


#=====================================================================
# MAIN
#=====================================================================
def main():
    args = parse_args()
    print(args)

    print('Hello')
    load_data = get_data_loader(args.data) 
    train_submission, args.preprocessing, args.skew, args.tangent = get_problem_functions(args.data)
    args.n_features, args.n_classes = get_data_shape(args.data)

    print('Building model ...')
    get_model = MODELS[args.model]
    model_args = extract_model_args(args, get_model)
    print( 'Model :', args.model)
    print( 'model_args :', model_args )
    model = get_model(**model_args)

    print('Loading data ...')
    X, y = load_data()

    print('Start training submission :', model.get_name())
    train_submission(model, X, y)

if __name__ == '__main__':
    main()