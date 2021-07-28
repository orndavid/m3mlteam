"""load_data.py
Load the dataset regarding Water recycle data project.
Before load is possible the data must exist in 
 >> ../data/IOTMeterData_new.csv
 This is created by downloading the dataset from kaggle and unzipping it
 """
import pandas as pd
import os


def __file_name():
    """
    Return the filename relative to the function call
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "../data", "IOTMeterData_new.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Could not find datafile")
    return file_path


def load_df():
    """Load the data as a dataframe. Must valuable when using
    jupyter lab
    """
    return pd.read_csv(__file_name())

