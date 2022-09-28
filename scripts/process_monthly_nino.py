import pandas as pd
import netCDF4 as nc
import xarray as xr

monthly_inds = []
for region in [12, 3, 34, 4]:
    ind = xr.open_dataarray(f'/scistor/ivm/jsn295/paper4/nino{region}_rel_monthly.nc', decode_times = False) # Mid stamped
    ind_series = ind.to_dataframe() # Lets do left stamping
    ind_series.index = pd.date_range(ind.time.attrs['units'][-10:-3], periods = len(ind), freq = 'MS') # Left stamping with first day of the month
    ind_series.columns = pd.MultiIndex.from_tuples([('nino',31,region,'mean')], names = ['variable','timeagg','clustid','metric'])
    monthly_inds.append(ind_series)

comb = pd.concat(monthly_inds, axis = 1).dropna()
comb.to_hdf('/scistor/ivm/jsn295/paper4/nino_rel_monthly.h5', key = 'nino')
