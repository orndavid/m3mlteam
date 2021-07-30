"""utils.py
Various utility scripts that can be used
through the entire package
"""
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


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

