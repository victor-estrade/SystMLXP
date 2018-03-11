#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse
import inspect

from problem.workflow import pprint

from models.mnist import mnist_models
from models.fashion_mnist import fashion_mnist_models
from models.higgsml import higsml_models
from models.higgsuci import higgsuci_models

from problem import mnist
from problem import fashion_mnist
from problem import higgs_geant
from problem import higgs_uci


MODELS = {
        'mnist': mnist_models,
        'fashion-mnist': fashion_mnist_models,
        'higgs-geant': higsml_models,
        'higgs-uci': higgsuci_models,
         }

ARG_MODELS = ['NN', 'ANN', 'TP', 'ATP', 'PAN', 'APAN', 'NNC', 'GB']

PROBLEMS = {
        'mnist': mnist,
        'fashion-mnist': fashion_mnist,
        'higgs-geant': higgs_geant,
        'higgs-uci': higgs_uci,
        }


def parse_args():
    # TODO : more descriptive msg.
    parser = argparse.ArgumentParser(
        description="Training launcher"
        )

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
        default=0, help="increase output verbosity")
    
    # DATASET CHOICE
    parser.add_argument('--data', help='chosen dataset',
        type=str, choices=PROBLEMS.keys(), default='mnist' )
    
    # MODEL CHOICE
    parser.add_argument('model', help='model to train',
        type=str, choices=ARG_MODELS )
    

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

    parser.add_argument('--n-augment', help='number of times the dataset is augmented',
        default=2, type=int)

    parser.add_argument('--n-adv-pre-training-steps', 
        help='number of update steps for the pre-training',
        default=3000, type=int)
    
    parser.add_argument('--n-clf-pre-training-steps', 
        help='number of update steps for the pre-training',
        default=3000, type=int)

    parser.add_argument('--n-recovery-steps', 
        help='number of update steps for the catch training of auxiliary models',
        default=5, type=int)

    parser.add_argument('--fraction-signal-to-keep', 
        help='fraction of signal to keep in Filters',
        default=0.95, type=float)


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


#=====================================================================
# stuff
#=====================================================================
def load_problem(data_name):
    problem = None
    if data_name in PROBLEMS:
        problem = PROBLEMS[data_name]
    else:
        raise ValueError('Unrecognized dataset name : {}'
                         'Expected one from {}'. format(data_name, PROBLEMS.keys()))
    return problem

def get_model_class(data_name, model_name):
    model_class = None
    if data_name in MODELS:
        model_class = MODELS[data_name](model_name)
    else:
        raise ValueError('Unrecognized dataset name : {}'
                         'Expected one from {}'. format(data_name, MODELS.keys()))
    return model_class

#=====================================================================
# MAIN
#=====================================================================
def main():
    args = parse_args()
    pprint(args)

    pprint('Hello')
    problem = load_problem(args.data)

    pprint('Building model ...')
    pprint( 'Model :', args.model)
    model_class = get_model_class(args.data, args.model)
    args.skewing_function = problem.skew
    args.tangent = problem.tangent
    model_args = extract_model_args(args, model_class)
    pprint( 'model_args :', model_args )
    model = model_class(**model_args)

    pprint('Loading data ...')
    X, y = problem.load_data()

    pprint('Start training submission :', model.get_name())
    problem.train_submission(model, X, y)


if __name__ == '__main__':
    main()