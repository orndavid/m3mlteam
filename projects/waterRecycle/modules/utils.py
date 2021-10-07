"""utils.py
Various utility scripts that can be used
through the entire package
"""
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from scipy import stats


def verify_name_in_series(df, y_name):
    """
    Verify that a column name is in the dataframe. Return a list
    of the x names without the y_name
    """
    x_names = list(df.columns)
    if y_name not in x_names:
        raise KeyError(f"Name : {y_name!r} not in dataframe")
    x_names.remove(y_name)
    return x_names


def rand_int_range():
    """
    Return a random integer within a predefined ragne. This
    is typically used for random seed values
    """
    return random.randint(0, 9999999)


def _cnorm(series):
    """Normalize a dataframe using the linear scaling"""
    return (series - np.min(series)) / (np.max(series) - np.min(series))


def normalize(method, df):
    """Normalize a dataframe using a specific method"""
    columns = df.columns
    if method == "range":
        return pd.DataFrame(data=MinMaxScaler().fit(df).transform(df),
                            columns=columns)
    elif method == "z":
        return stats.zscore(df)
    elif method == "maxabs":
        return pd.DataFrame(data=MaxAbsScaler().fit(df).transform(df),
                            columns=columns)
