# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import json

import numpy as np
import pandas as pd


def filter_arrays(idx, *arrays):
    filtered_arrays = tuple(arr[idx] if arr is not None else None for arr in arrays)
    return filtered_arrays

def filter_pandarrays(idx, *dataframes):
    filtered_df = tuple(df.iloc[idx].copy() if df is not None else None for df in dataframes)
    return filtered_df

class ClassifierFilter(object):
    def __init__(self, clf, fraction_signal_to_keep=0.95):
        super().__init__()
        self.clf = clf
        self.fraction_signal_to_keep =  fraction_signal_to_keep
        self.score_threshold_ = 0

    def fit(self, X, y, sample_weight=None):
        if isinstance(y, pd.core.generic.NDFrame):
            y = y.values
        proba = self.clf.predict_proba(X)
        clf_score = proba[:, 1]
        idx = np.argsort(clf_score)
        if sample_weight is not None:
            if isinstance(sample_weight, pd.core.generic.NDFrame):
                sample_weight = sample_weight.values
            fraction_signals_kept = np.cumsum(sample_weight[idx] * y[idx]) / np.sum(sample_weight * y)
        else:
            fraction_signals_kept = np.cumsum(y[idx]) / np.sum(y)
        i = np.searchsorted(fraction_signals_kept, 1-self.fraction_signal_to_keep)
        self.score_threshold_ = float(clf_score[idx[i]])
        return self

    def filter_idx(self, X):
        proba = self.clf.predict_proba(X)
        clf_score = proba[:, 1]
        idx = np.argsort(clf_score)
        i = np.searchsorted(clf_score[idx], self.score_threshold_)
        return idx[i:]

    def filter(self, X, *arrays):
        idx =  self.filter_idx( X )
        # TODO : Probably a cleaner way of sampling from DataFrame and NumpyArrays
        if isinstance(X, pd.core.generic.NDFrame):
            return filter_pandarrays( idx, X, *arrays )
        else:
            return filter_arrays( idx, X, *arrays )

    def __call__(self, X, *arrays):
        return self.filter(X, *arrays)

    def save_state(self, path):
        with open(path, 'w') as f:
            data = dict(score_threshold=self.score_threshold_, 
                       )
            json.dump(data, f)

    def load_state(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.score_threshold_ = data['score_threshold']

