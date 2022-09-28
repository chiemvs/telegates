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

    
if __name__ == '__main__':
    sstindex = makeindex(deseason = True, remove_interannual=False, timeagg = 21, degree = 7)
    response = create_response(respagg = 31, degree = 7)
    
    separation = -15
    idxname = '21d_deseas_d7'
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
        basedir = Path('~/paper4/anomalies/full/')
        extension = '_anom'
    elif anom_20:
        basedir = Path('~/paper4/anomalies/last20/')
        extension = '_anom20yr'
    else:
        basedir = Path('~/ERA5')
        extension = ''
    
    for compvar in compvars:
    
        var = compvar.split('_')[0]
        randid = uuid.uuid4().hex
    
        filepath = basedir / f'{compvar}{".anom.deg7" if (anom or anom_20) else ""}.nc'
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
