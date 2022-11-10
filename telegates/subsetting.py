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

def digitize_rolling(combined_frame : pd.DataFrame, quantiles : list = [0.33,0.66], nyearslice: int = 21, overall_thresholds: pd.DataFrame = None, record_counts: bool = False):
    """
    No fixed thresholds for t2m-mean-anom, overall thresholds can be presupplied
    """
    centeryears = combined_frame.index.year.unique().sort_values()[(nyearslice//2):-(nyearslice//2)]
    print(centeryears)
    which_category = combined_frame.copy().astype(int)
    if overall_thresholds is None:
        overall_thresholds = combined_frame.quantile(quantiles)
    if record_counts:
        overall = count_combinations(digitized_array=digitize(combined_frame, thresholds=overall_thresholds))
        hidden_counts = pd.DataFrame(np.nan, index = centeryears, columns = overall.stack(0).index)

    for i, centeryear in enumerate(centeryears):
        roll_sl = slice(pd.Timestamp(f'{centeryear - nyearslice//2}-01-01'),pd.Timestamp(f'{centeryear + nyearslice//2}-12-31'))
        roll_comb = combined_frame.loc[roll_sl,:] # combined contains only summer values.
        thresholds = overall_thresholds.copy()
        thresholds.loc[:,'t2m-mean-anom'] = roll_comb.quantile(quantiles).loc[:,'t2m-mean-anom'] # replacing only the t2m threshold
        roll_categories = digitize(roll_comb, thresholds=thresholds)
        if i == 0: # Filling from zeroth enry onwards
            which_category.loc[slice(None,roll_categories.index.max()),:] = roll_categories
        else: # Otherwise from central year until the end
            which_category.loc[slice(f'{centeryear}-01-01',roll_categories.index.max()),:] = roll_categories.loc[slice(f'{centeryear}-01-01',None),:] # replacing centeryear and beyond
        if record_counts:
            count = count_combinations(digitized_array=roll_categories)
            hidden_counts.loc[centeryear,:] = count.stack(0)

    if record_counts:
        return which_category, hidden_counts
    else:
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
    grouped = splitted.groupby(splitted.index.names[:-1])
    means = grouped.mean()
    stds = grouped.std()
    mins = grouped.min()
    maxs = grouped.max()
    return means.unstack(1), stds.unstack(1), mins.unstack(1), maxs.unstack(1)

def split_index(digitized_frame: pd.DataFrame, to_split: pd.Series):
    """
    Split another series into subsets by means of the digitized combinations
    Groups of combinations (e.g. exceedence in one combined with exceedence in other)
    are extracted and for each the appropriate proportion of the to_split index is returned
    """
    splitted = split(digitized_frame = digitized_frame, to_split = to_split)
    timestamps = splitted.groupby(splitted.index.names[:-1]).apply(lambda s: s.index.get_level_values('time'))
    return timestamps

