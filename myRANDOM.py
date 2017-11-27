# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

#=====================================================================
# Define some model
#=====================================================================
from sklearn.base import BaseEstimator, ClassifierMixin

class RANDOM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_class=2):
        super().__init__()
        self.n_class = n_class

    def fit(self, X, y):
        self.n_class = len(np.unique(y))
        return self

    def predict(self, X):
        return np.random.randint(0, self.n_class+1, size=X.shape[0])

    def predict_proba(self, X):
        proba = np.random.uniform(0, 1, size=(X.shape[0], self.n_class))
        proba = proba / np.sum(proba, axis=1)
        return proba

    def save(self, path):
        pass

    def load(self, path):
        pass


def get_model(n_classes=2):
    model = RANDOM(n_class=n_classes)
    return model
