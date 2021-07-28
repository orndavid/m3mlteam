"""main.py
Created:  Wed 28 Jul 2021 18:56:57 AEST
Objective:
    This is the entry point to testing various methods. Once the environment
    has been setup correctly with pip

    >> pip install -r requirements.txt
    
    The script is called from the terminal 

    >> python main.py (*args if applicable)

    The business logic is defined in modules/

    The data source is a kaggle dataset.
"""
import os

from modules.system import ensure_config
from modules.load_data import load_df


if __name__ == '__main__':
    # First step is to ensure the system has the correct 
    # configurations
    ensure_config()
    df = load_df()
    print(df.head())

