import os
import sys
import uuid
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

regions = pd.DataFrame({'latrange':[slice(-1.5,5.5),slice(10.75,15.25),slice(19.5,24.25)],
    'lonrange':[slice(162,169),slice(147,151.75),slice(155,161)]},
    index = pd.Index(['warm1','cold1','cold2']), dtype = 'object')

eradir = Path(os.path.expanduser('~/ERA5/'))
    
def selectregion(array: xr.DataArray, name: str):
    assert (name in regions.index), f'choose one of the region names in {regions.index}'
    return array.sel(latitude = regions.loc[name,'latrange'], longitude = regions.loc[name,'lonrange'])

def spatial_mean(array: xr.DataArray):
    stacked = array.stack({'latlon':['latitude','longitude']})
    return stacked.mean('latlon')

def agg_time(array: xr.DataArray, ndayagg: int = 1, method: str = 'mean') -> xr.DataArray:
    """
    Aggegates a daily time dimension by rolling averaging,
    Time axis should be continuous, otherwise non-neighbouring values are taken together. 
    It returns a left stamped aggregation of ndays
    Trailing Nan's are removed.
    """
    assert (np.diff(array.time) == np.timedelta64(1,'D')).all(), "time axis should be a continuous daily to be aggregated, though nan is allowed"

    name = array.name
    attrs = array.attrs
    f = getattr(array.rolling({'time':ndayagg}, center = False), method) # Stamped right
    array = f()
    array = array.assign_coords(time = array.time - pd.Timedelta(str(ndayagg - 1) + 'D')).isel(time = slice(ndayagg - 1, None)) # Left stamping, trailing nans removed
    array.name = name
    array.attrs = attrs
    return array

def fit_poly(array, degree: int = 3, year: int = None) -> xr.DataArray:
    """
    Fit 3rd order seasonal polynomial for each gridpoint in this data. 
    X is day of the year.
    Possibly for a single year, otherwise all years are joined
    If fitted per year discontinuities around winter are more likely.
    returns array of coefficients.
    """
    if not (year is None):
        array = array.sel(time = (warm.time.dt.year == year))
    X = array.coords['time'].dt.dayofyear
    y = array.stack({'latlon':['latitude','longitude']})
    coefs = np.polynomial.polynomial.polyfit(x = X, y = y, deg = degree)
    coefs = xr.DataArray(coefs, dims = ('coefs','latlon'), coords = y.coords['latlon'].coords)
    coefs = coefs.assign_coords({'coefs':np.arange(degree+1)})
    return coefs.unstack('latlon')

def evaluate_poly(array : xr.DataArray, coefs: xr.DataArray, year: int = None):
    """
    Evaluates the polynomial on the first time dimension. np.Polyval cannot do this for all gridcells at once
    So custom computation.
    """
    if not (year is None):
        array = array.sel(time = (warm.time.dt.year == year))
    X = array.coords['time'].dt.dayofyear
    y = xr.DataArray(np.zeros(array.shape), coords = array.coords, dims = array.dims)
    for degree in range(len(coefs)): # possibly later: https://en.wikipedia.org/wiki/Horner%27s_method
        y += X**degree * coefs.sel(coefs = degree, drop = True)
    return y

def deseasonalize(array: xr.DataArray, per_year: bool = False, return_polyval: bool = False, degree = 3):
    """
    If per year then trend is removed (and likely also interannual variability)
    plus you'll get a jump on the first of january. (not so important for summer)
    """
    deseasonalized = array.copy()
    deseasonalized.name = f'{deseasonalized.name}-anom'
    if per_year:
        years = np.unique(array.time.dt.year)
        if return_polyval:
            polyval = array.copy()
        for year in years:
            yearly_polyval = evaluate_poly(array, coefs = fit_poly(array, year = year, degree = degree), year = year)
            deseasonalized.loc[deseasonalized.time.dt.year == year,...] = deseasonalized.loc[deseasonalized.time.dt.year == year,...] - yearly_polyval
            if return_polyval:
                polyval.loc[polyval.time.dt.year == year,...] = yearly_polyval
    else:
        polyval = evaluate_poly(array, coefs = fit_poly(array, degree = degree))
        deseasonalized -= polyval
    if return_polyval:
        return deseasonalized, polyval
    else:
        return deseasonalized

def makeindex(deseason = True, remove_interannual = True, timeagg: int = None):
    """ 
    Whether to deseason on the daily timescale and per gridpoint
    Remove interannual only relevant if deasonalizing
    Aggregation by time possible before constructing the index (left stamping)
    """
    with xr.open_dataarray(eradir / 'sst_nhplus.nc') as da:
        warm = selectregion(da,'warm1')
        cold1 = selectregion(da,'cold1')
        cold2 = selectregion(da,'cold2')
        components = {'w':warm,'c1':cold1,'c2':cold2}
    if deseason:
        for key,field in components.items():
            components[key] = deseasonalize(field,per_year=remove_interannual, return_polyval=False, degree = 4)
    if not (timeagg is None):
        for key,field in components.items():
            components[key] = agg_time(array = field, ndayagg = timeagg)
    #index = spatial_mean(components['w']) - np.stack([spatial_mean(components['c1']),spatial_mean(components['c2'])]).min(axis = 0)
    index = spatial_mean(components['w']) - np.stack([spatial_mean(components['c1']),spatial_mean(components['c2'])]).mean(axis = 0)
    return index
            
def lag_precursor(precursor: pd.Series, separation: int, timeagg: int):
    """timeagg = timeagg of the precursor"""
    lagged = precursor.copy()
    lagged.index = lagged.index + pd.Timedelta(abs(separation) + timeagg, unit = 'D')
    return lagged

def create_response(respagg = 31):
    subdomainlats = slice(40,56)
    subdomainlons = slice(-5,24)
    t2m = xr.open_dataarray(eradir / 't2m_europe.nc').sel(latitude = subdomainlats, longitude = subdomainlons)
    clusterfield = xr.open_dataarray('/scistor/ivm/jsn295/clusters/t2m-q095.nc').sel(nclusters = 15, latitude = subdomainlats, longitude = subdomainlons)
    t2manom = deseasonalize(t2m,per_year=False, return_polyval=False, degree = 5)
    spatmean = t2manom.groupby(clusterfield).mean('stacked_latitude_longitude')
    spatmean = spatmean.sel(clustid = 9)
    response = agg_time(spatmean, ndayagg = respagg)
    return response

def combine_index_response(idx, idxname, response: xr.DataArray, lag = True, separation = -15, only_months: list = None, detrend_response = False):
    """
    If summeronly then only summer values are used for detrending.
    """
    if not (type(idx) in [pd.Series, pd.DataFrame]):
        ids = idx.to_pandas()
        ids.name = idx.name
    else:
        ids = idx
    if idxname.startswith('21'):
        precursoragg = 21
    else:
        precursoragg = 1
    if lag:
        lagged = lag_precursor(ids, separation = separation, timeagg = precursoragg)
        lagged.name = ids.name
    else:
        lagged = ids
    response_pd = response.to_pandas()
    response_pd.name = response.name
    combined = pd.merge(lagged, response_pd, left_index = True, right_index = True, how = 'inner')
    if not (only_months is None):
        combined = combined.loc[combined.index.month.map(lambda m: m in only_months),:]
    if detrend_response:
        combined.loc[:,response.name] = detrend(combined.loc[:,response.name])
    return combined
    
def digitize(combined_frame : pd.DataFrame, thresholds: pd.DataFrame):
    """
    shapes: frame (n_samples, n_variables), thresholds (nthresholds, n variables)
    """
    which_category = combined_frame.copy()
    for var in combined_frame.columns:
        which_category[var] = np.digitize(combined_frame[var], thresholds[var])
    return which_category

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

def split_index(digitized_frame: pd.DataFrame, to_split: pd.Series):
    """
    Split another series into subsets by means of the digitized combinations
    Groups of combinations (e.g. exceedence in one combined with exceedence in other)
    are extracted and for each the appropriate proportion of the to_split index is returned
    """
    splitted = split(digitized_frame = digitized_frame, to_split = to_split)
    timestamps = splitted.groupby(splitted.index.names[:-1]).apply(lambda s: s.index.get_level_values('time'))
    return timestamps

if __name__ == '__main__':
    sstindex = makeindex(deseason = True, remove_interannual=False, timeagg = 21)
    response = create_response(respagg = 31)
    
    separation = -15
    idxname = '21d_deseas_inter'
    combined = combine_index_response(idx = sstindex, idxname = idxname, response = response,
                                      lag = True, separation = separation, only_months = [6,7,8], detrend_response = False)
    
    timeslices = [slice('2000-01-01','2021-01-01')]
    
    compvars = ['sst_nhplus','u300_nhnorm','v300_nhnorm','z300_nhnorm','olr_tropics','stream_nhnorm']
    #compvars = ['t2m_europe']
    anom = False
    anom_20 = True
    timeagg = False
    
    assert not (anom_20 and anom), 'If you choose to deseason on last 20 years you cannot pick the anomalies'
    
    outdir = Path('~/paper4/analysis/composites')
    
    if anom:
        basedir = Path('~/paper4/anomalies/')
        extension = '_anom'
    elif anom_20:
        basedir = Path('~/paper4/anomalies20/')
        extension = '_anom20yr'
    else:
        basedir = Path('~/ERA5')
        extension = ''
    
    for compvar in compvars:
    
        var = compvar.split('_')[0]
        randid = uuid.uuid4().hex
    
        filepath = basedir / f'{compvar}{".anom.deg4" if (anom or anom_20) else ""}.nc'
        da = xr.open_dataarray(filepath)
        timestamps= da.coords['time'].to_pandas()
    
        for sl in timeslices:
            comb = combined.loc[sl,:]
            thresholds = comb.quantile([0.5])
            digits = digitize(comb, thresholds=thresholds)
            ts = timestamps.loc[sl]
            subsets = split_index(digits, ts)
            comps = []
            if timeagg: # Time aggregation with cdo. to 21 days
                da.close()
                stdout = os.system(f'cdo --timestat_date first -P 10 runmean,21 -seldate,{sl.start},{(pd.Timestamp(sl.stop)+pd.Timedelta(21,"d")).strftime("%Y-%m-%d")} {str(filepath)} ~/temp_{randid}.nc')
                da = xr.open_dataarray(f'~/temp_{randid}.nc', drop_variables = 'time_bnds')
            for ind in subsets:
                laggedind = ind - pd.Timedelta(abs(separation), unit='D')
                comps.append(da.sel(time = laggedind).mean('time'))
            comps = xr.concat(comps,dim = subsets.index).unstack('concat_dim')
            comps.attrs.update({'nsamples':str(subsets.index.names) + str(subsets.map(lambda a: len(a)).to_dict()), 'units':da.units})
            outname = f'{sl.start}_{sl.stop}_{compvar}{extension}{"_21d" if timeagg else ""}.nc'
            comps.to_netcdf(outdir / outname)
            da.close()
