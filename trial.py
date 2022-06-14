import os
import sys
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

eradir = Path(os.path.expanduser('~/ERA5/'))

#da = xr.open_dataarray(eradir / 'olr_tropics.nc')
da = xr.open_dataset(eradir / 'sst_nhplus.nc', group = 'mean')['sst-mean']

regions = pd.DataFrame({'latrange':[slice(-1.5,5.5),slice(10.75,15.25),slice(19.5,24.25)],
    'lonrange':[slice(162,169),slice(147,151.75),slice(155,161)]},
    index = pd.Index(['warm1','cold1','cold2']), dtype = 'object')

def selectregion(array: xr.DataArray, name: str):
    assert (name in regions.index), f'choose one of the region names in {regions.index}'
    return array.sel(latitude = regions.loc[name,'latrange'], longitude = regions.loc[name,'lonrange'])

warm = selectregion(da,'warm1')
cold1 = selectregion(da,'cold1')
cold2 = selectregion(da,'cold2')

def spatial_mean(array: xr.DataArray):
    stacked = array.stack({'latlon':['latitude','longitude']}) 
    return stacked.mean('latlon')

def annual_mean(array: xr.DataArray):
    return array.groupby(array.time.dt.year).mean()

index = spatial_mean(warm) - np.stack([spatial_mean(cold1),spatial_mean(cold2)]).min(axis = 0) 

# Third order polynomial?, if per year then discontinuities?
#X = warm.coords['time'].to_index().to_julian_date()
X = warm.coords['time'].dt.dayofyear
y = warm.stack({'latlon':['latitude','longitude']})
coefs = np.polynomial.polynomial.polyfit(x = X, y = y, deg = 3)
p = np.polynomial.polynomial.polyval(x = X[:365], c = coefs[:,:2], tensor = False)
