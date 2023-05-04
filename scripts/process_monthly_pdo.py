import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

"""
replacing PDO index, formerly UW-JISAO (with global trend removed), now ERSST
About this way of defining see also:
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014GL061992
"""

dataurl = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_PDO.nc?time%2CPDO%2Cindex&time%3E=1950-01-01&time%3C=2023-04-01"
temppath = Path('/scistor/ivm/jsn295/paper4/cciea_OC_PDO.nc')
finalpath = Path('/scistor/ivm/jsn295/paper4/pdo_monthly.h5')
oldpdopath = Path('/scistor/ivm/jsn295/paper4/oldpdo/pdo_monthly.h5')

if not temppath.exists():
    print('downloading original netcdf')
    os.system(f'curl {dataurl} -o {temppath}')

if not finalpath.exists():
    array = xr.open_dataset(temppath)
    print('creating pdo hdf')
    newpdo = pd.Series(array.PDO.values, index = pd.DatetimeIndex(array.time.values), name = 'PDO')
    newpdo.to_hdf(finalpath, key = 'pdo')
else:
    print('reading pdo hdf')
    newpdo = pd.read_hdf(finalpath)
oldpdo = pd.read_hdf(oldpdopath)
comparison = pd.concat([oldpdo, newpdo], axis = 1, join = 'inner') 

