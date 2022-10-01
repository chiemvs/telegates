import os
import sys
import uuid
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/telegates/'))

from telegates.processing import deseasonalize, makeindex, create_response, combine_index_response
from telegates.subsetting import digitize, split, split_index

TMPDIR = Path(sys.argv[1]) # Should enter as a string
COMPVAR = sys.argv[2] # Should enter as a string, one of ['sst_nhplus','u300_nhnorm','v300_nhnorm','z300_nhnorm','olr_tropics','stream_nhnorm']
ANOM = sys.argv[3] # Whether to composite anomalies (and what type)
assert ANOM in ['full','last20','False'], 'anom should be full, last20 or False'
AGGTIME = sys.argv[4].lower() == 'true' # Whether to aggregate daily values before compositing

basepath = Path(os.path.expanduser('~/paper4/'))
outdir = basepath / 'analysis' / 'composites'
    
if __name__ == '__main__':
    respagg = 31
    separation = -15
    idxagg = 21
    degree = 7
    fullname = f't2m_{respagg}_sst_{idxagg}_sep_{separation}_deg_{degree}.h5' 
    fullpath = basepath / 'analysis' / fullname
    if not fullpath.exists():
        print('creating combined dataframe')
        sstindex = makeindex(deseason = True, remove_interannual=False, timeagg = 21, degree = degree)
        response = create_response(respagg = respagg, degree = degree)
    
        combined = combine_index_response(idx = sstindex, idxname = f'{idxagg}d_sst', response = response,
                                      lag = True, separation = separation, only_months = [6,7,8], detrend_response = False)
        combined.to_hdf(fullpath, key = 'combined')
    else:
        print('combined dataframe directly loaded')
        combined = pd.read_hdf(fullpath)
    
    timeslices = [slice('1950-01-01','1966-01-01'),slice('1968-01-01','1983-01-01'),slice('2000-01-01','2021-01-01')]
    
    if ANOM == 'False':
        datadir = Path(os.path.expanduser('~/ERA5'))
        dataname = f'{COMPVAR}.nc'
    else:
        datadir = basepath / 'anomalies' / ANOM 
        dataname = f'{COMPVAR}.anom.deg{degree}.nc'

    assert (datadir / dataname).exists()
    
    var = COMPVAR.split('_')[0]
    
    print(f'cp {str(datadir / dataname)} {str(TMPDIR / dataname)}') # Copy to faster directory.
    os.system(f'cp {str(datadir / dataname)} {str(TMPDIR / dataname)}') # Copy to faster directory.
    da = xr.open_dataarray(TMPDIR / dataname)
    timestamps= da.coords['time'].to_pandas()
    
    for sl in timeslices:
        comb = combined.loc[sl,:]
        thresholds = comb.quantile([0.5])
        digits = digitize(comb, thresholds=thresholds)
        ts = timestamps.loc[sl]
        subsets = split_index(digits, ts)
        comps = []
        if AGGTIME: # Time aggregation with cdo. to 21 days
            randfile = uuid.uuid4().hex + '.nc'
            da.close()
            print(f'cdo --timestat_date first -P 10 runmean,{idxagg} -seldate,{sl.start},{(pd.Timestamp(sl.stop)+pd.Timedelta(idxagg,"d")).strftime("%Y-%m-%d")} {str(TMPDIR / dataname)} {str(TMPDIR / randfile)}')
            os.system(f'cdo --timestat_date first -P 10 runmean,{idxagg} -seldate,{sl.start},{(pd.Timestamp(sl.stop)+pd.Timedelta(idxagg,"d")).strftime("%Y-%m-%d")} {str(TMPDIR / dataname)} {str(TMPDIR / randfile)}')
            da = xr.open_dataarray(TMPDIR / randfile, drop_variables = 'time_bnds')
        for ind in subsets:
            laggedind = ind - pd.Timedelta(abs(separation), unit='D')
            comps.append(da.sel(time = laggedind).mean('time'))
        comps = xr.concat(comps,dim = subsets.index).unstack('concat_dim')
        comps.attrs.update({'nsamples':str(subsets.index.names) + str(subsets.map(lambda a: len(a)).to_dict()), 'units':da.units})
        outname = f'{sl.start}_{sl.stop}_{COMPVAR}_anom{ANOM}{"_" + str(idxagg) if AGGTIME else ""}.nc'
        print('writing: ', outname)
        comps.to_netcdf(outdir / outname, mode = 'w')
        da.close()
