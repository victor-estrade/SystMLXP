# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


from .neural_network_model import NeuralNetModel
from .neural_network_model import AugmentedNeuralNetModel
from .neural_network_model import BlindNeuralNetModel

from .tangent_prop_model import TangentPropModel
from .tangent_prop_model import AugmentedTangentPropModel

from .pivot_model import PivotModel
from .pivot_model import AugmentedPivotModel

from .cascade_model import CascadeNeuralNetModel

from .gradient_boost import GradientBoostingModel
from .gradient_boost import BlindGradientBoostingModel

MODELS = {
    'NN': NeuralNetModel,
    'ANN': AugmentedNeuralNetModel,
    'BNN': BlindNeuralNetModel,
    'TP': TangentPropModel,
    'ATP': AugmentedTangentPropModel,
    'PAN': PivotModel,
    'APAN': AugmentedPivotModel,
    'NNC': CascadeNeuralNetModel,
    'GB': GradientBoostingModel,
    'BGB': BlindGradientBoostingModel,
}


def higsml_models(model_name):
    if model_name in MODELS:
        return MODELS[model_name]
    else:
        raise ValueError('Unrecognized model name : {}'
                         'Expected one from {}'. format(model_name, MODELS.keys()))
