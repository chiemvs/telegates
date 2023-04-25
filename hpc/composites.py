import os
import sys
import uuid
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/telegates/'))

from telegates.processing import deseasonalize, makeindex, makeindex2, create_response, combine_index_response
from telegates.subsetting import digitize, digitize_rolling, split, split_index

TMPDIR = Path(sys.argv[1]) # Should enter as a string
COMPVAR = sys.argv[2] # Should enter as a string, one of ['sst_nhplus','u300_nhnorm','v300_nhnorm','z300_nhnorm','olr_tropics','stream_nhnorm']
ANOM = sys.argv[3] # Whether to composite anomalies (and what type)
assert ANOM in ['full','last20','last40','False'], 'anom should be full, last20, last40 or False'
AGGTIME = sys.argv[4].lower() == 'true' # Whether to aggregate daily values before compositing
ROLLTHRESH = sys.argv[5].lower() == 'true' # Whether the t2m terciles are computed with a rolling threshold (useful for long timeslices)
BOOTSTRAP = sys.argv[6].lower() == 'true' # Whether to block bootstrap samples

basepath = Path(os.path.expanduser('~/paper4/'))
analysisdir = basepath / 'analysis' / 'dipole' # Whether the dipole or tripole index is used
    
if __name__ == '__main__':
    respagg = 28
    separation = -15
    wpdagg = 28
    predagg = 28
    degree = 7
    outdir = analysisdir / f'composites_t2m_{respagg}_sst_{wpdagg}_sep_{separation}_deg_{degree}'
    fullname = f't2m_{respagg}_sst_{wpdagg}_sep_{separation}_deg_{degree}.h5' 
    fullpath = analysisdir / fullname
    if not fullpath.exists():
        print('creating combined dataframe')
        sstindex = makeindex2(deseason = True, remove_interannual=False, timeagg = wpdagg, degree = degree)
        response = create_response(respagg = respagg, degree = degree)
    
        combined = combine_index_response(idx = sstindex, idxname = f'{wpdagg}d_sst', response = response,
                                      lag = True, separation = separation, only_months = [6,7,8], detrend_response = False)
        combined.to_hdf(fullpath, key = 'combined')
    else:
        print('combined dataframe directly loaded')
        combined = pd.read_hdf(fullpath)
    
    quantiles = [0.3333,0.6666]
    overall_thresholds = combined.quantile(quantiles)
    #timeslices = [slice('1968-01-01','1983-01-01'),slice('2000-01-01','2021-08-31')]
    timeslices = [slice('1980-01-01','2021-10-01')]
    
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
        if ROLLTHRESH: # Rolling thresholds for t2m
            digits = digitize_rolling(comb, quantiles = quantiles, nyearslice=21,
                    overall_thresholds = overall_thresholds, record_counts = False)
        else:
            thresholds = overall_thresholds.copy() # Varying the t2m threshold, but not the west pacific threshold
            thresholds.loc[:,'t2m-mean-anom'] = comb.quantile(quantiles).loc[:,'t2m-mean-anom'] # replacing only the t2m threshold, for the global warming effect
            digits = digitize(comb, thresholds=thresholds)
        ts = timestamps.loc[sl]
        subsets = split_index(digits, ts)
        comps = []
        if AGGTIME: # Time aggregation with cdo. to 21 days
            randfile = uuid.uuid4().hex + '.nc'
            da.close()
            print(f'cdo --timestat_date first -P 10 runmean,{predagg} -seldate,{(pd.Timestamp(sl.start)-pd.Timedelta(predagg + abs(separation),"d")).strftime("%Y-%m-%d")},{(pd.Timestamp(sl.stop)+pd.Timedelta(predagg,"d")).strftime("%Y-%m-%d")} {str(TMPDIR / dataname)} {str(TMPDIR / randfile)}')
            os.system(f'cdo --timestat_date first -P 10 runmean,{predagg} -seldate,{(pd.Timestamp(sl.start)-pd.Timedelta(predagg + abs(separation),"d")).strftime("%Y-%m-%d")},{(pd.Timestamp(sl.stop)+pd.Timedelta(predagg,"d")).strftime("%Y-%m-%d")} {str(TMPDIR / dataname)} {str(TMPDIR / randfile)}')
            da = xr.open_dataarray(TMPDIR / randfile, drop_variables = 'time_bnds')
        for ind in subsets: # Subsets are the categorized indices (9 times)
            moments = {'preinit': ind - pd.Timedelta(abs(separation) + predagg, unit='D'),
                    'postinit':ind - pd.Timedelta(abs(separation), unit='D'),
                    'valid':ind}
            subcomps = []
            if not BOOTSTRAP: # Insert block bootstrapping here?
                for moment, moment_index in moments.items():
                    subcomps.append(da.sel(time = moment_index).mean('time').expand_dims({'moment':[moment]}))
                ind_fields = xr.concat(subcomps, dim = 'moment')
            else: # Block bootstrapping from entire population mean of same length. same sample regardless of moment
                present_years = digits.index.year.unique()
                for i in range(500):
                    years_drawn = present_years[np.random.randint(0,len(present_years),len(present_years))]
                    sample_stamps = pd.DatetimeIndex([])
                    j = 0
                    while len(sample_stamps) < len(ind): # No fixed length of ind therefore adaptive sampling
                        sample_stamps = sample_stamps.append(digits.index[digits.index.year == years_drawn[j]])
                        j += 1

                    subcomps.append(da.sel(time = sample_stamps).mean('time').expand_dims({'draw':[i]}))
                ind_fields = xr.concat(subcomps, dim = 'draw')
            comps.append(ind_fields)
        comps = xr.concat(comps,dim = subsets.index).unstack('concat_dim')
        comps.attrs.update({'nsamples':str(subsets.index.names) + str(subsets.map(lambda a: len(a)).to_dict()), 'units':da.units})
        outname = f'{sl.start}_{sl.stop}_{COMPVAR}_anom{ANOM}{"_" + str(predagg) if AGGTIME else ""}{"_rollthresh" if ROLLTHRESH else ""}{"_bootstrapped" if BOOTSTRAP else ""}.nc'
        print('writing: ', outname)
        comps.to_netcdf(outdir / outname, mode = 'w')
        da.close()
