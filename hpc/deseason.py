import os
import sys
import itertools
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/Weave/'))

from Weave.inputoutput import Writer

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

def deseasonalize(array: xr.DataArray, per_year: bool = False, return_polyval: bool = False, degree = 3, normalslice: slice = slice(None)):
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


def make_anoms(name: str, blocksize = 50, degree: int = 3, normalslice: slice = slice(None)):
    """
    Normalslice can be used to determine the period that is the normal
    if not supplied the polynomials are computed on all available timesteps
    """
    varname = name.split('_')[0]
    try:
        da = xr.open_dataarray(datadir / f'{name}.nc')
    except ValueError:
        da = xr.open_dataset(datadir / f'{name}.nc')[varname] # Multiple variables in the file
    outpath = anomdir / f'{name}.anom.deg{degree}.nc'

    wr = Writer(datapath = outpath, varname = varname)
    wr.create_dataset(example = da)


    latstarts = np.arange(0,len(da.coords['latitude']),blocksize) 
    lonstarts = np.arange(0,len(da.coords['longitude']),blocksize) 
    
    for latstartidx, lonstartidx in itertools.product(latstarts,lonstarts):
    
        latendidx = latstartidx + blocksize # Not inclusive
        if latendidx > latstarts.max():
            latendidx = None
        latslice = slice(latstartidx,latendidx)
    
        lonendidx = lonstartidx + blocksize # Not inclusive
        if lonendidx > lonstarts.max():
            lonendidx = None
        lonslice = slice(lonstartidx,lonendidx)
    
        temparr = da.isel(latitude = latslice, longitude = lonslice)
        
        deseasonalized = deseasonalize(temparr, normalslice = normalslice) 
    
        wr.write_spatial_block(array = deseasonalized, latslice = latslice, lonslice = lonslice, units = da.units )

datadir = Path(os.path.expanduser('~/ERA5/'))
anomdir = Path(os.path.expanduser('~/paper4/anomalies20'))
normalslice = slice('2000-01-01','2021-01-01')

for name, deg in zip(['v300_nhnorm','u300_nhnorm','z300_nhnorm','sst_nhplus','olr_tropics'],[4,4,4,4,4]):
    make_anoms(name = name, degree = deg, blocksize = 100, normalslice = normalslice)
#make_anoms(name = 't2m_europe', degree = 5, blocksize = 150, normalslice = normalslice)
make_anoms(name = 'stream_nhnorm', degree = 4, blocksize = 150, normalslice = normalslice)
