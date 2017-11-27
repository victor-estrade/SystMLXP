# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pandas as pd
import numpy as np

from merging import merge_abscisse_fill_idx

__doc__ = "Help function to computes the various metrics."
__version__ = "1.0"
__author__ = "Victor Estrade"


def sort_decision_y_W(decision, y_true, weights):
    """
    Ascending order (ie from small to big decission values)
    """
    decision = np.asarray(decision)
    y_true = np.asarray(y_true)
    weights = np.asarray(weights)
    idx = np.argsort(decision)
    sorted_decision = decision[idx]
    sorted_y_true = y_true[idx]
    sorted_weights = weights[idx]
    return sorted_decision, sorted_y_true, sorted_weights

def merge_abscisse(*abscisses):
    """
    Merge the given abscisses.
    TODO : improve explaination
    
    Returns merged_abscisse, indexes
    -------
        merged_abscisse : (numpy.array) merged abscisses
        indexes : (list of numpy.array) a list of indexes.
            Each element of the list gives the indexes to extrapolate values using the right side following each given abscisse
    """
    INT = np.int
    FLOAT = np.float
    n_abscisses = len(abscisses)
    if n_abscisses < 2 :
        raise ValueError('Need at least 2 abscisse arrays to merge : {} given'.format(n_abscisses))
    concat_abscisse = np.concatenate(abscisses, axis=0)
    merged_abscisse = np.unique(concat_abscisse)
    size = merged_abscisse.shape[0]
    indexes = [np.empty(size, dtype=INT) for _ in range(n_abscisses)]
    for idx, absc in zip(indexes, abscisses):
        merge_abscisse_fill_idx(np.asarray(absc, dtype=FLOAT), idx, np.asarray(merged_abscisse, dtype=FLOAT))
    return merged_abscisse, indexes


def bining(abscisse, n_bin, min_value=0., max_value=1.0):
    """
    Produce bin and return the indexes corresponding to the given abscisse.
    """
    bins = np.linspace(min_value, max_value, n_bin)
    idx = np.empty(n_bin, dtype=np.int32)
    j = 0
    for i in range(abscisse.shape[0]):
        if j < n_bin-1 and bins[j] < abscisse[i]:
            idx[j] = i
            j += 1
    while j < n_bin:
        idx[j] = i
        j += 1
    return bins, idx


def integrated_true_positive_rate(sorted_y_true, sorted_weights, positive_label=1):
    """
    Integration from right to left.
    """
    weighted_true_positive_rate = (sorted_y_true==positive_label)*sorted_weights
    return np.cumsum(weighted_true_positive_rate[::-1])[::-1]


def integrated_false_positive_rate(sorted_y_true, sorted_weights, negative_label=0):
    """
    Integration from right to left.
    """
    weighted_false_positive_rate = (sorted_y_true==negative_label)*sorted_weights
    return np.cumsum(weighted_false_positive_rate[::-1])[::-1]


def AMS3(true_positive_rate, false_positive_rate):
    """
    Computes the AMS $\frac{s}{\sqrt(b)}$
    with :
        s = true_positive_rate
        b = false_positive_rate
    """
    return true_positive_rate / np.sqrt( false_positive_rate)

def AMS2(true_positive_rate, false_positive_rate):
    """
    Computes the AMS2 $\sqrt{ 2 \left ( (s+b) ln (1 + \frac{s}{b}) - s \right ) }$
    with :
        s = true_positive_rate
        b = false_positive_rate
    """
    s = true_positive_rate
    b = false_positive_rate
    return np.sqrt( 2 * ( (s+b) * np.log( 1 + s / b ) - s ) )

def AMS1(true_positive_rate, false_positive_rate, false_positive_rate_z):
    """
    Computes the AMS1 $\sqrt( 2 * ( (s+b) * ln( (s + b)/b_0 ) - s -b - b_0 ) )$
    with :
        s = true_positive_rate
        b = false_positive_rate
        b_0 = 0.5 * \left ( b - \sigma_b^2 + \sqrt{ (b - \sigma_b^2)^2 + 4 * (s+b) * \sigma_b^2 } \right )
    """
    s = true_positive_rate
    b = false_positive_rate
    sigma_b = np.abs(false_positive_rate_z - false_positive_rate) + 1e-7
    b_0 = 0.5 * ( b - sigma_b**2 + np.sqrt( (b - sigma_b**2)**2 + 4 * (s+b) * (sigma_b**2) ) )
    return np.sqrt( 2 * ( (s+b) * np.log( (s + b)/b_0 ) - s - b + b_0 ) + ( ( b - b_0 ) / sigma_b )**2 )

def error_stat(true_positive_rate, false_positive_rate):
    """
    Computes the statistic error $\frac{\sqrt(s+b)}{s}$
    with :
        s = true_positive_rate
        b = false_positive_rate
    """
    # Safely compute and set to numerical values results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        res = np.true_divide( np.sqrt( true_positive_rate + false_positive_rate ), true_positive_rate )
        # Replace inf values with arbitrary value
        # The value should not be the min or max. 
        # Choose mean because it is fast to compute.
        res[res == np.inf] = np.mean(res)
        res = np.nan_to_num(res)
    return res

def error_syst(s, b, s0, b0):
    """
    Computes :
    $$
    \left( \frac{ s+b-b[TES=0] }{ s[TES=0] } - 1 \right)
    $$
    the systematic error

    Parameters
    ----------
        s: the true positive rate of the skewed data
        b: the false positive rate of the skewed data
        s0: the true positive rate of the original data
        b0: the false positive rate of the original data
    Return
    ------
        err : (numpy array) the syst error values
    """
    # Safely compute and set to numerical values results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        tmp = np.true_divide( ( s + b - b0 ), s0 )
        # Replace inf values with arbitrary value
        # The value should not be the min or max. 
        # Choose mean because it is fast to compute.
        tmp[tmp == np.inf] = np.mean(tmp)
        tmp = np.nan_to_num(tmp)
    return np.abs( tmp - 1 )

def sigma_mu(s, b, s0, b0):
    """
    Computes :
    $$
    \sigma_\mu = \sqrt{ \left( \frac{\sqrt{s[TES=0]+b[TES=0]}}{s[TES=0]} \right)^2 + \left( \frac{ s+b-b[TES=0] }{ s[TES=0] } - 1 \right)^2 }
    $$

    Parameters
    ----------
        s: the true positive rate of the skewed data
        b: the false positive rate of the skewed data
        s0: the true positive rate of the original data
        b0: the false positive rate of the original data
    Return
    ------
        sigma_mu : (numpy array) the sigma_mu values
    """
    return np.sqrt( error_stat(s0, b0)**2 + error_syst(s,b,s0,b0)**2 )
