import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr

def digitize(combined_frame : pd.DataFrame, thresholds: pd.DataFrame):
    """
    shapes: frame (n_samples, n_variables), thresholds (nthresholds, n variables)
    """
    which_category = combined_frame.copy()
    for var in combined_frame.columns:
        which_category[var] = np.digitize(combined_frame[var], thresholds[var])
    return which_category

def count_combinations(digitized_array, normalize: bool = False):
    """
    Digitized array has e.g. 0,1,2 values for three terciles.
    returns count matrix. If normalized then frequencies (count/total)
    """
    nclasses = np.unique(digitized_array).max() + 1
    counts = pd.DataFrame(np.zeros((nclasses,nclasses)), 
                          index = pd.RangeIndex(nclasses, name = digitized_array.columns[0]), 
                          columns = pd.RangeIndex(nclasses, name = digitized_array.columns[1]))
    for i in range(len(digitized_array)):
        a,b = digitized_array.iloc[i,:]
        counts.iloc[a,b] += 1
    if normalize:
        counts = counts/counts.sum().sum()
    return counts

def split(digitized_frame: pd.DataFrame, to_split: pd.Series):
    """
    Split another series into subsets by means of the digitized combinations
    Groups of combinations (e.g. exceedence in one combined with exceedence in other)
    are extracted. Properties of the other series are computed with timestamps of the groups.
    """
    match = digitized_frame.index.intersection(to_split.index)
    if len(match) != len(digitized_frame.index):
        warnings.warn(f'to split series is {len(to_split.index) - len(digitized_frame.index)} longer than digitized combinations, only {match.min()} till {match.max()} can be used')
    splitted = digitized_frame.groupby(digitized_frame.columns.to_list()).apply(lambda x: to_split.loc[x.index.intersection(to_split.index)])
    return splitted

def split_compute(digitized_frame: pd.DataFrame, to_split: pd.Series):
    """
    Split another series into subsets by means of the digitized combinations
    Groups of combinations (e.g. exceedence in one combined with exceedence in other)
    are extracted. Properties of the other series are computed with timestamps of the groups.
    """
    splitted = split(digitized_frame = digitized_frame, to_split = to_split)
    means = splitted.groupby(splitted.index.names[:-1]).mean()
    stds = splitted.groupby(splitted.index.names[:-1]).std()
    return means.unstack(1), stds.unstack(1)

def split_index(digitized_frame: pd.DataFrame, to_split: pd.Series):
    """
    Split another series into subsets by means of the digitized combinations
    Groups of combinations (e.g. exceedence in one combined with exceedence in other)
    are extracted and for each the appropriate proportion of the to_split index is returned
    """
    splitted = split(digitized_frame = digitized_frame, to_split = to_split)
    timestamps = splitted.groupby(splitted.index.names[:-1]).apply(lambda s: s.index.get_level_values('time'))
    return timestamps

