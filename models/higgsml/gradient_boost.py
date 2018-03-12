# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


class GradientBoostingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, n_estimators=1000, max_depth=3,):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = GradientBoostingClassifier(learning_rate=learning_rate,
                                 n_estimators=n_estimators, 
                                 max_depth=max_depth,
                                 )

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        if isinstance(y, pd.core.generic.NDFrame):
            y = y.values
        if isinstance(sample_weight, pd.core.generic.NDFrame):
            sample_weight = sample_weight.values
        self.clf.fit(X, y, sample_weight=sample_weight)


    def predict(self, X):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        y_pred = self.clf.predict(X)
        return y_pred

    def predict_proba(self, X):
        if isinstance(X, pd.core.generic.NDFrame):
            X = X.values
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, dir_path):
        """Save the model in the given directory"""
        path = os.path.join(dir_path, 'GradientBoosting.pkl')
        joblib.dump(self.clf, path)
        return self

    def load(self, dir_path):
        """Load the model of th i-th CV from the given directory"""
        path = os.path.join(dir_path, 'GradientBoosting.pkl')
        self.clf = joblib.load(path)
        return self

    def get_name(self):
        name = "GradientBoostingModel-{}-{}-{}".format(self.learning_rate, self.n_estimators, self.max_depth)
        return name
