import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

sl = '1980-01-01_2021-10-01'
degree = 7
respagg = 28
wpdagg = 28
separation = -15
basedir = Path('/scistor/ivm/jsn295/paper4/analysis/dipole/')
datapath = basedir / f'composites_t2m_{respagg}_sst_{wpdagg}_sep_{separation}_deg_{degree}'

alphalevel = 0.1

available = list(datapath.glob(f'{sl}*_bootstrapped.nc'))

for filepath in available:

    outpath = filepath.parent / '_'.join(filepath.name.split('_')[:-1] + [f'q{alphalevel}.nc']) 
    if not outpath.exists():
        print('trying to create ', outpath)
        with xr.open_dataarray(filepath) as draws:
            nsamples = eval(draws.attrs['nsamples'].split(']')[-1])
            # straightforward diff
            pos_pos = draws.sel({'sst-mean-anom':2, 't2m-mean-anom':2}) 
            pos_pos_samples = nsamples[(2,2)]
            neg_neg = draws.sel({'sst-mean-anom':0, 't2m-mean-anom':0})
            diff = pos_pos - neg_neg
            diff_qs = diff.quantile(q = [alphalevel/2,1-alphalevel/2], dim='draw')
            diff_qs.attrs = draws.attrs
            diff_qs.attrs.update({'alpha':alphalevel, 'computation':'(2,2)-(0,0)'})

            # Weighted diff
            pos_neut = draws.sel({'sst-mean-anom':2, 't2m-mean-anom':1}) 
            pos_neut_samples = nsamples[(2,1)]
            pos_neg = draws.sel({'sst-mean-anom':2, 't2m-mean-anom':0}) 
            pos_neg_samples = nsamples[(2,0)]
            weighted_diff = pos_pos - (pos_neut * pos_neut_samples + pos_neg * pos_neg_samples)/(pos_neut_samples + pos_neg_samples)
            weighted_diff_qs = weighted_diff.quantile(q = [alphalevel/2,1-alphalevel/2], dim='draw')
            weighted_diff_qs.attrs = draws.attrs
            weighted_diff_qs.attrs.update({'alpha':alphalevel, 'computation':'(2,2)-[(2,1),(2,0)]'})

        total = xr.Dataset({'diff':diff_qs,'weighted_diff':weighted_diff_qs})

        total.to_netcdf(outpath)
    else:
        print(outpath, ' already exists')

