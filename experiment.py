# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import numpy as np

from higgs_metrics import sort_decision_y_W
from higgs_metrics import merge_abscisse
from higgs_metrics import integrated_true_positive_rate
from higgs_metrics import integrated_false_positive_rate
from higgs_metrics import AMS1
from higgs_metrics import AMS2
from higgs_metrics import AMS3
from higgs_metrics import sigma_mu
from higgs_metrics import error_stat
from higgs_metrics import error_syst


__doc__ = "Help function to computes the various metrics on many runs, merge them and reduce them to mean and variance."
__version__ = "1.0"
__author__ = "Victor Estrade"


def decision_TP_FP(decision, y, W):
    """
    sort, integrate and reduce with unique.
    """
    sorted_decision, sorted_y, sorted_W = sort_decision_y_W(decision, y, W)
    TP = integrated_true_positive_rate(sorted_y, sorted_W)
    FP = integrated_false_positive_rate(sorted_y, sorted_W)
    unique_decision, uniq_idx = np.unique(sorted_decision, return_index=True)
    TP = TP[uniq_idx]
    FP = FP[uniq_idx]
    return unique_decision, TP, FP


def bining_constant_density(one_df, n_bin=5000):
    if np.unique(one_df['decision']).shape[0] < n_bin:
        new_df = one_df
    else :
        idx = np.arange(0, one_df.shape[0], one_df.shape[0]//n_bin)
        new_df = one_df.iloc[idx].reset_index().copy()
    return new_df


def integrated_base_metrics(decision, y, W):
    sorted_decision, sorted_y, sorted_W =  sort_decision_y_W(decision, y, W)
    TP = integrated_true_positive_rate(sorted_y, sorted_W)
    FP = integrated_false_positive_rate(sorted_y, sorted_W)
    # Fix me : Can't remember why taking only unique points was a mistake ?
    # unique_decision, uniq_idx = np.unique(sorted_decision, return_index=True)
    # TP = TP[uniq_idx]
    # FP = FP[uniq_idx]
    base_metrics = pd.DataFrame({'decision': sorted_decision, 'TP':TP, 'FP': FP})
    return base_metrics


def basic_metrics_run(run, n_bin=5000):
    if n_bin is None:
        metric_run = {sysTES: integrated_base_metrics( df['decision'], df['Label'], df['Weight'] )
            for sysTES, df in run.items()
        }
    else:
        metric_run = {sysTES: bining_constant_density( integrated_base_metrics( df['decision'], df['Label'], df['Weight'] ), n_bin=n_bin )
            for sysTES, df in run.items()
        }
    return metric_run


def basic_metrics_xp(xp, n_bin=5000):
    metric_xp = [basic_metrics_run(run, n_bin=n_bin) for run in xp]
    return metric_xp


def complete_metrics(base_metrics):
    # TODO add more metrics
    base_metrics['error_stat'] = error_stat(base_metrics['TP'], base_metrics['FP'])
    base_metrics['AMS2'] = AMS2(base_metrics['TP'], base_metrics['FP'])
    base_metrics['AMS3'] = AMS3(base_metrics['TP'], base_metrics['FP'])
    return base_metrics


def complete_metrics_run(run):
    complete_run = {sysTES: complete_metrics(df) for sysTES, df in run.items()}
    return complete_run


def complete_metrics_xp(xp):
    complete_xp = [complete_metrics_run(run) for run in xp]
    return complete_xp


def merge_metrics_decision(metrics):
    merged_decision, indexes = merge_abscisse(*[m['decision'] for m in metrics])
    result = []
    for m, idx in zip(metrics, indexes):
        new = pd.DataFrame({'decision': merged_decision})
        for c in m.columns.drop(['decision']):
            new[c] = m[c].values[idx]
        result.append(new)
    return result

def merge_decision_xp(xp):
    all_df = merge_metrics_decision([df for run in xp for df in run.values() ])
    idx = np.cumsum([0] + [len(run.values()) for run in xp])
    new_xp = [{ sysTES : df for sysTES, df in zip(run.keys(), all_df[start:stop]) } 
              for run, start, stop in zip( xp, idx[:-1], idx[1:] ) ]
    return new_xp

def systematic_metrics(base_metrics, base_metrics_training):
    # TODO add more metrics ?
    s = base_metrics['TP']
    b = base_metrics['FP']
    s_0 = base_metrics_training['TP']
    b_0 = base_metrics_training['FP']
    base_metrics['error_syst'] = error_syst(s, b, s_0, b_0)
    base_metrics['sigma_mu'] = sigma_mu(s, b, s_0, b_0)
    base_metrics['AMS1'] = AMS1(s_0, b_0, b)
    return base_metrics

def systematic_run(run, training_TES):
    systematic_run = {sysTES: systematic_metrics(df, run[training_TES]) 
                for sysTES, df in run.items()}
    return systematic_run


def systematic_xp(xp, training_TES):
    systematic_xp = [systematic_run(run, training_TES) for run in xp]
    return systematic_xp


def reduce_mean_xp(xp):
    sysTES_set = set( [sysTES for run in xp for sysTES in run.keys()] )
    columns = set( [col for run in xp for df in run.values() for col in df.columns] )
    columns.remove('decision')
    xp_mean = {}
    for sysTES in sysTES_set:        
        mean_df = pd.DataFrame({'decision': xp[0][sysTES]['decision']})
        for col in columns:
            mean_df[col] = np.mean(pd.concat([run[sysTES][col] for run in xp if col in run[sysTES].columns], axis=1), axis=1)
        xp_mean[sysTES] = mean_df
    return xp_mean

def reduce_std_xp(xp):
    sysTES_set = set( [sysTES for run in xp for sysTES in run.keys()] )
    columns = set( [col for run in xp for df in run.values() for col in df.columns] )
    columns.remove('decision')
    xp_std = {}
    for sysTES in sysTES_set:        
        std_df = pd.DataFrame({'decision': xp[0][sysTES]['decision']})
        for col in columns:
            std_df[col] = pd.concat([run[sysTES][col] for run in xp if col in run[sysTES].columns], axis=1).std(axis=1)
        xp_std[sysTES] = std_df
    return xp_std
