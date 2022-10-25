import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import detrend
from pathlib import Path

from .utils import select_months

regions = pd.DataFrame({'latrange':[slice(-1.5,5.5),slice(10.75,15.25),slice(19.5,24.25)],
    'lonrange':[slice(162,169),slice(147,151.75),slice(155,161)]},
    index = pd.Index(['warm1','cold1','cold2']), dtype = 'object')

regions2 = pd.DataFrame({'latrange':[slice(-1.5,5.5),slice(10,14)],
    'lonrange':[slice(162,169),slice(139.5,150.5)]},
    index = pd.Index(['warm1','cold1']), dtype = 'object')

eradir = Path(os.path.expanduser('~/ERA5/'))

def selectregion(array: xr.DataArray, name: str, select_from: pd.DataFrame = regions):
    assert (name in select_from.index), f'choose one of the region names in {select_from.index}'
    if select_from.loc[name,'lonrange'].stop < select_from.loc[name,'lonrange'].start: # Crossing the dateline
        interval = float(np.unique(np.diff(array.longitude)))
        lonrange = np.concatenate([np.arange(select_from.loc[name,'lonrange'].start,180,interval),
            np.arange(-180,select_from.loc[name,'lonrange'].stop,interval)])
    else:
        lonrange = select_from.loc[name,'lonrange']
    return array.sel(latitude = select_from.loc[name,'latrange'], longitude = lonrange)

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

def deseasonalize(array: xr.DataArray, per_year: bool = False, return_polyval: bool = False, degree = 4, normalslice: slice = slice(None)):
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
        polyval = evaluate_poly(array, coefs = fit_poly(array.sel(time = normalslice), degree = degree))
        deseasonalized -= polyval
    if return_polyval:
        return deseasonalized, polyval
    else:
        return deseasonalized

def makeindex2(deseason = True, remove_interannual = True, timeagg: int = None, degree: int = 4):
    """ 
    Whether to deseason on the daily timescale and per gridpoint
    Remove interannual only relevant if deasonalizing
    Aggregation by time possible before constructing the index (left stamping)
    """
    with xr.open_dataarray(eradir / 'sst_nhplus.nc') as da:
        warm = selectregion(da,'warm1', select_from = regions2)
        cold1 = selectregion(da,'cold1', select_from = regions2)
        components = {'w':warm,'c1':cold1}
    if deseason:
        for key,field in components.items():
            components[key] = deseasonalize(field,per_year=remove_interannual, return_polyval=False, degree = degree)
    if not (timeagg is None):
        for key,field in components.items():
            components[key] = agg_time(array = field, ndayagg = timeagg)
    #index = spatial_mean(components['w']) - np.stack([spatial_mean(components['c1']),spatial_mean(components['c2'])]).min(axis = 0)
    index = spatial_mean(components['w']) - spatial_mean(components['c1'])
    return index

def makeindex(deseason = True, remove_interannual = True, timeagg: int = None, degree: int = 4):
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
            components[key] = deseasonalize(field,per_year=remove_interannual, return_polyval=False, degree = degree)
    if not (timeagg is None):
        for key,field in components.items():
            components[key] = agg_time(array = field, ndayagg = timeagg)
    #index = spatial_mean(components['w']) - np.stack([spatial_mean(components['c1']),spatial_mean(components['c2'])]).min(axis = 0)
    index = spatial_mean(components['w']) - np.stack([spatial_mean(components['c1']),spatial_mean(components['c2'])]).mean(axis = 0)
    return index
            
def lag_precursor(precursor: pd.Series, separation: int, timeagg: int, sign: bool = False):
    """
    timeagg = timeagg of the precursor
    Can be made sign-sensitive, for both forward and backward lagging. 
    With forward lagging the separation is relative to the beginning of response.
    """
    lagged = precursor.copy()
    if sign and (separation > 0):
        lagged.index = lagged.index - pd.Timedelta(separation, unit = 'D') 
    else:
        lagged.index = lagged.index + pd.Timedelta(abs(separation) + timeagg, unit = 'D')
    return lagged

def create_response(respagg: int = 31, degree: int = 5):
    subdomainlats = slice(40,56)
    subdomainlons = slice(-5,24)
    t2m = xr.open_dataarray(eradir / 't2m_europe.nc').sel(latitude = subdomainlats, longitude = subdomainlons)
    clusterfield = xr.open_dataarray('/scistor/ivm/jsn295/clusters/t2m-q095.nc').sel(nclusters = 15, latitude = subdomainlats, longitude = subdomainlons)
    t2manom = deseasonalize(t2m,per_year=False, return_polyval=False, degree = degree)
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
    precursoragg = int(idxname.split('_')[0][:-1])
    if lag:
        print(f'lagging, agg: {precursoragg}, sep: {separation}')
        lagged = lag_precursor(ids, separation = separation, timeagg = precursoragg)
        lagged.name = ids.name
    else:
        lagged = ids
    response_pd = response.to_pandas()
    response_pd.name = response.name
    combined = pd.merge(lagged, response_pd, left_index = True, right_index = True, how = 'inner')
    if not (only_months is None):
        combined = select_months(df = combined, months = only_months)
    if detrend_response:
        combined.loc[:,response.name] = detrend(combined.loc[:,response.name])
    return combined
