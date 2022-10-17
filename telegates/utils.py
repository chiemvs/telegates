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
