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

sys.path.append(os.path.expanduser('~/Documents/telegates/'))

from telegates.processing import deseasonalize

TMPDIR = Path(sys.argv[1]) # Should enter as a string
NAME = sys.argv[2] # Should enter as a string

# Copy the right data to tempdir
print(f'cp {os.path.expanduser("~/ERA5/")}{NAME}.nc {TMPDIR / NAME}.nc')
os.system(f'cp {os.path.expanduser("~/ERA5/")}{NAME}.nc {TMPDIR / NAME}.nc')

def make_anoms(name: str, outpath: Path, blocksize = 50, degree: int = 4, normalslice: slice = slice(None)):
    """
    Normalslice can be used to determine the period that is the normal
    if not supplied the polynomials are computed on all available timesteps
    """
    varname = name.split('_')[0]
    try:
        da = xr.open_dataarray(TMPDIR / f'{name}.nc')
    except ValueError:
        da = xr.open_dataset(TMPDIR / f'{name}.nc')[varname] # Multiple variables in the file

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
        
        deseasonalized = deseasonalize(temparr, degree = degree, normalslice = normalslice) 
    
        wr.write_spatial_block(array = deseasonalized, latslice = latslice, lonslice = lonslice, units = da.units )
    wr.write_attrs(attrs = {'degree':degree})

anomdir = Path(os.path.expanduser('~/paper4/anomalies'))
for normalslice, subdir  in zip([slice(None)],['full']): #zip([slice('2000-01-01','2021-01-01'), slice(None)], ['last20','full']):
    for degree in [5,7]:
        tmpoutpath = TMPDIR / f'{NAME}.anom.deg{degree}.{subdir}.nc'
        outpath = anomdir / subdir / f'{NAME}.anom.deg{degree}.nc'
        make_anoms(name = NAME, outpath = tmpoutpath, degree = degree, blocksize = 120, normalslice = normalslice)
        print(f'mv {tmpoutpath} {outpath}')
        os.system(f'mv {tmpoutpath} {outpath}')
