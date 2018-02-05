#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

__doc__="""
ATLAS Higgs Machine Learning Challenge 2014
Read CERN Open Data Portal Dataset http://opendata.cern.ch/record/328
and manipulate it
 - Label is changd from charcter to integer 0 or 1
 - DetailLabel is introduced indicating subpopulations
 - systematics effect are simulated
     - bkg_weight_norm : manipulates the background weight
     - tau_energy_scale : manipulates PRI_tau_pt and recompute other quantities accordingly
             Some WARNING : variable DER_mass_MMC is not properly manipulated (modification is linearised), 
             and I advocate to NOT use DER_mass_MMC when doSystTauEnergyScale is enabled
             There is a threshold in the original HiggsML file at 20GeV on PRI_tau_energy. 
             This threshold is moved when changing sysTauEnergyScale which is unphysicsal. 
             So if you're going to play with sysTauEnergyScale (within 0.9-1.1), 
             I suggest you remove events below say 22 GeV *after* the manipulation
             applying doSystTauEnergyScale with sysTauENergyScale=1. does NOT yield identical results as not applyield 
             doSystTauEnergyScale, this is because of rounding error and zero mass approximation.
             doSysTauEnerbyScale impacts PRI_tau_pt as well as PRI_met and PRI_met_phi
    - so overall I suggest that when playing with doSystTauEnergyScale, the reference is
          - not using DER_mass_MMC
          - applying *after* this manipulation PRI_tau_pt>22
          - run with sysTauENergyScale=1. to have the reference
          
Author D. Rousseau LAL, Nov 2016

Modification Dec 2016 (V. Estrade):
- Wrap everything into separated functions.
- V4 class now handle 1D-vector values (to improve computation efficiency).
- Fix compatibility with both python 2 and 3.
- Use pandas.DataFrame to ease computation along columns
- Loading function for the base HiggsML dataset (fetch it on the internet if needed)

Refactor March 2017 (V. Estrade):
- Split load function (cleaner)
"""
__version__ = "0.1"
__author__ = "David Rousseau, and Victor Estrade "

import os
# import sys
# import gzip
import copy

import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit

from .download import maybe_download
from .download import get_data_dir

from .workflow import pprint
from .workflow import check_dir
from .workflow import _get_save_directory

COLUMN_NAMES = {
    0: 'PRI_lep_1_pt',
    1: 'PRI_lep_1_eta',
    2: 'PRI_lep_1_phi',
    3: 'PRI_lep_2_pt',
    4: 'PRI_lep_2_eta',
    5: 'PRI_lep_2_phi',
    6: 'PRI_met',
    7: 'PRI_met_phi',
    8: '8',
    9: '9',
    10: 'relative_MET',
    11: 'axial_MET',
    12: 'MMC',
    13: 'delta_R',
    14: 'delta_eta',
    15: '15',
    16: '16',
    17: 'Pt/Pt',
    18: 'invariant_mass_visible',
    19: 'invariant_mass_ll',
    20: 'delta_phi',
    21: 'sphericity',
    22: 'transverse_sphericity',
    23: '23',
    24: '24',
    }

RESTRICTED_COLUMNS = [0,1,2,3,4,5,6,7]

COLUMN_RENAME_FOR_SKEWING = {
    'PRI_lep_1_pt': 'PRI_tau_pt',
    'PRI_lep_1_eta': 'PRI_tau_eta',
    'PRI_lep_1_phi': 'PRI_tau_phi',
    'PRI_lep_2_pt': 'PRI_lep_pt',
    'PRI_lep_2_eta': 'PRI_lep_eta',
    'PRI_lep_2_phi': 'PRI_lep_phi',
    }

def load_htautau(nrows=None, restricted_cols=True):
    url = "http://mlphysics.ics.uci.edu/data/htautau/htautau.txt.gz"
    filename = os.path.join(get_data_dir(), "htautau.txt.gz")
    maybe_download(filename, url)
    if restricted_cols :
        data = pd.read_csv(filename, sep='\t', nrows=nrows, header=None, usecols=RESTRICTED_COLUMNS)
    else:
        data = pd.read_csv(filename, sep='\t', nrows=nrows, header=None)
    data.rename(columns=COLUMN_NAMES, inplace=True)
    return data

def load_ztautau(nrows=None, restricted_cols=True):
    url = "http://mlphysics.ics.uci.edu/data/htautau/ztautau.txt.gz"
    filename = os.path.join(get_data_dir(), "ztautau.txt.gz")
    maybe_download(filename, url)
    if restricted_cols :
        data = pd.read_csv(filename, sep='\t', nrows=nrows, header=None, usecols=RESTRICTED_COLUMNS)
    else:
        data = pd.read_csv(filename, sep='\t', nrows=nrows, header=None)
    data.rename(columns=COLUMN_NAMES, inplace=True)
    return data


def load_higgstautau(n_samples=None):
    if n_samples is None:
        data_h = load_htautau()
        data_z = load_ztautau()
    else:
        data_h = load_htautau(nrows=n_samples//2)
        data_z = load_ztautau(nrows=n_samples//2)

    data_h["Label"] = np.ones(data_h.shape[0])
    data_z["Label"] = np.zeros(data_z.shape[0])
    data_h["Weight"] = np.ones(data_h.shape[0])/200
    data_z["Weight"] = np.ones(data_z.shape[0])
    data = pd.concat([data_h, data_z])
    data.reset_index(inplace=True, drop=True)
    return data

def load_data():
    data = load_higgstautau(n_samples=10000000)
    data.rename(columns=COLUMN_RENAME_FOR_SKEWING, inplace=True)
    tau_energy_scale(data, 1.0)
    return data, None



# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================

class V4:
    """
    A simple 4-vector class to ease calculation
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self,apx=0., apy=0., apz=0., ae=0.):
        """
        Constructor with 4 coordinates
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
        if self.e + 1e-3 < self.p():
            raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

    def copy(self):
        return copy.deepcopy(self)
    
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2
    
    def p(self):
        return np.sqrt(self.p2())
    
    def pt2(self):
        return self.px**2 + self.py**2
    
    def pt(self):
        return np.sqrt(self.pt2())
    
    def m(self):
        return np.sqrt( np.abs( self.e**2 - self.p2() ) ) # abs is needed for protection
    
    def eta(self):
        return np.arcsinh( self.pz/self.pt() )
    
    def phi(self):
        return np.arctan2(self.py, self.px)
    
    def deltaPhi(self, v):
        """delta phi with another v"""
        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """delta eta with another v"""
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """delta R with another v"""
        return np.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """recompute e given m"""
        return np.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):
        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*np.cos(phi)
        self.py = pt*np.sin(phi)
        self.pz = pt*np.sinh(eta)
        self.e = self.eWithM(m)
    
    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e
    
    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px += other.px
            self.py += other.py
            self.pz += other.pz
            self.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self
    
    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = self.copy()
        try:
            copy.px += other.px
            copy.py += other.py
            copy.pz += other.pz
            copy.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

# magic variable
# FIXME : does it really returns sqrt(2) if in dead center ?
def METphi_centrality(aPhi, bPhi, cPhi):
    """
    Calculate the phi centrality score for an object to be between two other objects in phi
    Returns sqrt(2) if in dead center
    Returns smaller than 1 if an object is not between
    a and b are the bounds, c is the vector to be tested
    """
    # Safely compute and set to zeros results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        
        A = np.true_divide(np.sin(cPhi - aPhi), np.sin(bPhi - aPhi))
        A[A == np.inf] = 0
        A = np.nan_to_num(A)
        
        B = np.true_divide(np.sin(bPhi - cPhi), np.sin(bPhi - aPhi))
        B[B == np.inf] = 0
        B = np.nan_to_num(B)
        
        res = (A+B)/np.sqrt(A**2 + B**2)
        res[res == np.inf] = 0
        res = np.nan_to_num(res)
    return res


# another magic variable
def eta_centrality(eta, etaJ1, etaJ2):
    """
    Calculate the eta centrality score for an object to be between two other objects in eta
    Returns 1 if in dead center
    Returns value smaller than 1/e if object is not between
    """
    center = (etaJ1 + etaJ2) / 2.
    
    # Safely compute and set to zeros results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore'):
        width  = 1. / (etaJ1 - center)**2
        width[width == np.inf] = 0
        width = np.nan_to_num(width)
        
    return np.exp(-width * (eta - center)**2)


# ==================================================================================
#  Now we enter in the manipulation procedures (everything works on data inplace)
# ==================================================================================

def label_to_float(data):
    """
    Transform the string labels to float values.
    s -> 1.0
    b -> 0.0

    Works inplace on the given data !

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
    """
    if data['Label'].dtype == object:
        #copy entry in human usable form
        data["Label"] = (data["Label"] == 's').astype("float")
    else:
        pass

# ==================================================================================
def getDetailLabel(origWeight, Label, num=True):
    """
    Given original weight and label, 
    return more precise label specifying the original simulation type.
    
    Args
    ----
        origWeight: the original weight of the event
        Label : the label of the event (can be {"b", "s"} or {0,1})
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.

    Return
    ------
        detailLabel: the corresponding detail label ("W" is the default if not found)

    Note : Could be better optimized but this is fast enough.
    """
    # prefer numeric detail label
    detail_label_num={
        57207:0, # Signal
        4613:1,
        8145:2,
        4610:3,
        917703: 105, #Z
        5127399:111,
        4435976:112,
        4187604:113,
        2407146:114,
        1307751:115,
        944596:122,
        936590:123,
        1093224:124,
        225326:132,
        217575:133,
        195328:134,
        254338:135,
        2268701:300 #T
        }
    # complementary for W detaillabeldict=200
    #previous alphanumeric detail label    
    detail_lable_str={
       57207:"S0",
       4613:"S1",
       8145:"S2",
       4610:"S3",
       917703:"Z05",
       5127399:"Z11",
       4435976:"Z12",
       4187604:"Z13",
       2407146:"Z14",
       1307751:"Z15",
       944596:"Z22",
       936590:"Z23",
       1093224:"Z24",
       225326:"Z32",
       217575:"Z33",
       195328:"Z34",
       254338:"Z35",
       2268701:"T"
    }

    if num:
        detailLabelDict = detail_label_num
    else:
        detailLabelDict = detail_label_str

    iWeight=int(1e7*origWeight+0.5)
    detailLabel = detailLabelDict.get(iWeight, "W") # "W" is the default value if not found
    if detailLabel == "W" and (Label != 0 and Label != 'b') :
        raise ValueError("ERROR! if not in detailLabelDict sould have Label==1 ({}, {})".format(iWeight,Label))
    return detailLabel

def add_detail_label(data, num=True):
    """
    Add a 'detailLabel' column with the detailed labels.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.
    """
    if "origWeight" in data.columns:
        detailLabel = [getDetailLabel(w, label, num=num) for w, label in zip(data["origWeight"], data["Label"])]
    else:
        detailLabel = [getDetailLabel(w, label, num=num) for w, label in zip(data["Weight"], data["Label"])]
    data["detailLabel"] = detailLabel

# ==================================================================================

def bkg_weight_norm(data, systBkgNorm):
    """
    Apply a scaling to the weight.
    Keeps the previous weights in the 'origWeight' columns

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    TODO maybe explain why it scales only the background.
    """
    # only a weight manipulation
    data["origWeight"] = data["Weight"]
    if not "detailLabel" in data.columns:
        add_detail_label(data)
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = ( data["Weight"]*systBkgNorm ).where(data["detailLabel"] == "W", other=data["origWeight"])

# ==================================================================================
# TES : Tau Energy Scale
# ==================================================================================

def tau_energy_scale(data, systTauEnergyScale):
    """
    Manipulate one primary input : the PRI_tau_pt and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        systTauEnergyScale : the factor applied : PRI_tau_pt <-- PRI_tau_pt * systTauEnergyScale

    Notes :
    -------
        Add 'ORIG_mass_MMC' and 'ORIG_sum_pt' columns.
        Recompute :
            - PRI_tau_pt
            - PRI_met
            - PRI_met_phi
            - DER_mass_transverse_met_lep
            - DER_mass_vis
            - DER_pt_h
            - DER_deltar_tau_lep
            - DER_pt_ratio_lep_tau
            - DER_met_phi_centrality
        Round up to 3 decimals.

    """
    # scale tau energy scale, arbitrary but reasonable value
    data["PRI_tau_pt"] *= systTauEnergyScale 

    # now recompute the DER quantities which are affected

    # first built 4-vectors
    vtau = V4() # tau 4-vector
    vtau.setPtEtaPhiM(data["PRI_tau_pt"], data["PRI_tau_eta"], data["PRI_tau_phi"], 0.8) # tau mass 0.8 like in original

    vlep = V4() # lepton 4-vector
    vlep.setPtEtaPhiM(data["PRI_lep_pt"], data["PRI_lep_eta"], data["PRI_lep_phi"], 0.) # lep mass 0 (either 0.106 or 0.0005 but info is lost)

    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(data["PRI_met"], 0., data["PRI_met_phi"], 0.) # met mass zero,

    # fix MET according to tau pt change
    vtauDeltaMinus = vtau.copy()
    vtauDeltaMinus.scaleFixedM( (1.-systTauEnergyScale)/systTauEnergyScale )
    vmet += vtauDeltaMinus
    vmet.pz = 0.
    vmet.e = vmet.eWithM(0.)
    data["PRI_met"] = vmet.pt()
    data["PRI_met_phi"] = vmet.phi()
 
    # compute many vector sum
    vtransverse = V4()
    vtransverse.setPtEtaPhiM(vlep.pt(), 0., vlep.phi(), 0.) # just the transverse component of the lepton
    vtransverse += vmet
    data["DER_mass_transverse_met_lep"] = vtransverse.m()

    vltau = vlep + vtau # lep + tau
    data["DER_mass_vis"] = vltau.m()

    vlmet = vlep + vmet # lep + met # Seems to be unused ?
    vltaumet = vltau + vmet # lep + tau + met

    data["DER_pt_h"] = vltaumet.pt()

    data["DER_deltar_tau_lep"] = vtau.deltaR(vlep)
    
    data["DER_pt_ratio_lep_tau"] = vlep.pt()/vtau.pt()


    data["DER_met_phi_centrality"] = METphi_centrality(data["PRI_lep_phi"], data["PRI_tau_phi"], data["PRI_met_phi"])

    # delete non trivial objects to save memory (useful?)
    # del vtau, vlep, vmet, vlmet, vltau, vltaumet

    # Fix precision to 3 decimals
    DECIMALS = 3
    
    data["PRI_tau_pt"] = data["PRI_tau_pt"].round(decimals=DECIMALS)
    data["PRI_tau_eta"] = data["PRI_tau_eta"].round(decimals=DECIMALS)
    data["PRI_tau_phi"] = data["PRI_tau_phi"].round(decimals=DECIMALS)
    data["PRI_lep_pt"] = data["PRI_lep_pt"].round(decimals=DECIMALS)
    data["PRI_lep_eta"] = data["PRI_lep_eta"].round(decimals=DECIMALS)
    data["PRI_lep_phi"] = data["PRI_lep_phi"].round(decimals=DECIMALS)
    data["PRI_met"] = data["PRI_met"].round(decimals=DECIMALS)
    data["PRI_met_phi"] = data["PRI_met_phi"].round(decimals=DECIMALS)
    data["DER_mass_transverse_met_lep"] = data["DER_mass_transverse_met_lep"].round(decimals=DECIMALS)
    data["DER_mass_vis"] = data["DER_mass_vis"].round(decimals=DECIMALS)
    data["DER_pt_h"] = data["DER_pt_h"].round(decimals=DECIMALS)
    data["DER_deltar_tau_lep"] = data["DER_deltar_tau_lep"].round(decimals=DECIMALS)
    data["DER_pt_ratio_lep_tau"] = data["DER_pt_ratio_lep_tau"].round(decimals=DECIMALS)
    data["DER_met_phi_centrality"] = data["DER_met_phi_centrality"].round(decimals=DECIMALS)


# ==================================================================================
#  WORKFLOW : 
# ==================================================================================

RANDOM_STATE = 42

def get_cv_iter(X, y):
    cv = ShuffleSplit(n_splits=12, test_size=0.2, random_state=RANDOM_STATE)
    cv_iter = list(cv.split(X, y))
    return cv_iter


def get_save_directory():
    dir = os.path.join( _get_save_directory(), 'higgs_uci')
    return check_dir(dir)


def split_data_label_weights(data):
    X = data.drop(["Weight", "Label"], axis=1)
    y = data["Label"]
    W = data["Weight"]
    return X, y, W

def normalize_weight(W, y, background_luminosity=410999.84732187376, signal_luminosity=691.9886077135781):
    """Normalize the given weight to assert that the luminosity is the same as the nominal.
    Returns the normalized weight vector/Series
    """
    background_weight_sum = W[y==0].sum()
    signal_weight_sum = W[y==1].sum()
    W_new = W.copy()
    W_new[y==0] = W[y==0] * ( background_luminosity / background_weight_sum )
    W_new[y==1] = W[y==1] * ( signal_luminosity / signal_weight_sum )
    return W_new

def split_train_test(data, idx_train, idx_test):
    n_samples = data.shape[0]
    n_train = idx_train.shape[0]
    n_test = n_samples - n_train
    if n_test < 0:
        raise ValueError('The number of train samples ({}) exceed the total number of samples ({})'.format(n_train, n_samples))
    train_data = data.iloc[idx_train].copy()
    test_data = data.iloc[idx_test].copy()
    # if 'Weight' in data.columns and 'Label' in data.columns:
    train_data['Weight'] = normalize_weight(train_data['Weight'], train_data['Label'])
    test_data['Weight'] = normalize_weight(test_data['Weight'], test_data['Label'])
    return train_data, test_data


def skew(data, sysTauEnergyScale=1.0, remove_mass_MMC=True):
    data_skewed = data.copy()
    if not "DER_mass_MMC" in data_skewed.columns:
        data_skewed["DER_mass_MMC"] =  np.zeros(data.shape[0]) # Add dummy column

    tau_energy_scale(data_skewed, sysTauEnergyScale)  # Modify data inplace
    
    if remove_mass_MMC and "DER_mass_MMC" in data_skewed.columns:
        data_skewed = data_skewed.drop( ["DER_mass_MMC"], axis=1 )
    return data_skewed

def cut(data, threshold=22.0):
    data_cut = data[data['PRI_tau_pt'] > threshold]
    return data_cut

def skewing_function(data, sysTauEnergyScale=1.0, remove_mass_MMC=True, threshold=22.0):
    skewed_data = skew(data, sysTauEnergyScale=sysTauEnergyScale, remove_mass_MMC=remove_mass_MMC)
    skewed_data = cut(skewed_data, threshold=threshold )
    return skewed_data

def tangent(df, alpha=1e-3):
    """ The approximate formula to get the tangent. """
    df_plus = skew(df, sysTauEnergyScale=1.0+alpha)
    df_minus = skew(df, sysTauEnergyScale=1.0-alpha)
    return ( df_plus - df_minus ) / ( 2 * alpha )

def balance_training_weight(w, y):
    """Balance the weights between positive and negative class."""
    sample_weight = w.copy()
    neg_mask = (y == 0)
    pos_mask = (y == 1)
    
    bkg_sum_weight = np.sum(sample_weight[neg_mask])
    sig_sum_weight = np.sum(sample_weight[pos_mask])

    sample_weight[pos_mask] = sample_weight[pos_mask] / sig_sum_weight
    sample_weight[neg_mask] = sample_weight[neg_mask] / bkg_sum_weight
    return sample_weight


def train_submission(model, data, y=None):
    X = data
    y = data['Label']
    cv_iter = get_cv_iter(X, y)
    n_cv = len(cv_iter)
    save_directory = get_save_directory()

    for i, (idx_dev, idx_valid) in enumerate(cv_iter):
        train_data, test_data =  split_train_test(data, idx_dev, idx_valid)
        train_data = skewing_function(train_data, sysTauEnergyScale=1.0)
        X, y, W = split_data_label_weights(train_data)
        W = balance_training_weight(W, y) * y.shape[0] / 2
        
        pprint('training {}/{}...'.format(i+1, n_cv))
        model.fit(X, y, sample_weight=W)

        pprint('saving model {}/{}...'.format(i+1, n_cv))
        model_name = '{}-{}'.format(model.get_name(), i)
        
        path = os.path.join(save_directory, model_name)
        check_dir(path)
        
        model.save(path)
    return None


def build_run(model, X, y, W, all_sysTES, skew_function):
    run = {}
    for sysTES in all_sysTES:
        X_skew = skew_function(X, sysTauEnergyScale=sysTES)
        indexes = X_skew.index
        proba = model.predict_proba(X_skew)
        n_samples_before = X.shape[0]
        n_samples_after = X_skew.shape[0]
        # W_skew = normalize_weight(W[indexes], y)
        run[sysTES] = pd.DataFrame({'decision': proba[:,1], 'Weight':W[indexes], 'Label': y[indexes]})
    return run


def test_submission(data, models, all_sysTES=(1.0, 1.03, 1.05, 1.1) ):
    X = data
    y = data['Label']
    cv_iter = get_cv_iter(X, y)
    n_cv = len(cv_iter)
    xp = []
    for i, (idx_dev, idx_valid) in enumerate(cv_iter):
        train_data, test_data =  split_train_test(data, idx_dev, idx_valid)
        X_test, y_test, W_test = split_data_label_weights(test_data)
        
        model = models[i]
        pprint('testing model {}/{}'.format(i+1, n_cv))
        run_i = build_run(model, X_test, y_test, W_test, all_sysTES, skew_function=skewing_function)
        xp.append(run_i)
    return xp
