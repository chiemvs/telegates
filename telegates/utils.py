import numpy as np
import pandas as pd

from typing import Union, List

from scipy.interpolate import interp1d

def interpolate(inp: Union[pd.DataFrame, pd.Series], kind: str = 'linear'):
    """
    Interpolation to daily values of all columns in a dataframe
    """
    if isinstance(inp, pd.Series):
        convert = True
        inp = inp.to_frame()
    else:
        convert = False
    interpolated = pd.DataFrame(np.nan, index = pd.date_range(inp.index.values.min(), inp.index.values.max(), freq = 'd'), columns = inp.columns)
    for key in inp.columns:
        f = interp1d(inp.index.to_julian_date(), inp.loc[:,key].values, kind = kind)
        interpolated.loc[:,key] = f(interpolated.index.to_julian_date())
    if convert:
        return interpolated.iloc[:,-1] # Returning as series again
    else:
        return interpolated


def select_months(df: Union[pd.DataFrame, pd.Series], months: List[int]):
    assert isinstance(df.index, pd.DatetimeIndex), 'pandas object needs to be datetime indexed, for selecting months'
    return df.loc[df.index.month.map(lambda m: m in months)]

def data_for_pcolormesh(array, shading:str):
    """Xarray array to usuable things"""
    lats = array.latitude.values # Interpreted as northwest corners (90 is in there)
    lons = array.longitude.values # Interpreted as northwest corners (-180 is in there, 180 not)
    if shading == 'flat':
        lats = np.concatenate([lats[[0]] - np.diff(lats)[0], lats], axis = 0) # Adding the sourthern edge 
        lons = np.concatenate([lons, lons[[-1]] + np.diff(lons)[0]], axis = 0)# Adding the eastern edge (only for flat shating)
    return lons, lats, array.values.squeeze()
