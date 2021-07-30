"""randomForest.py
Wrapper for creating a random forest system
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

from tqdm import tqdm

from modules.utils import verify_name_in_series
from modules.utils import rand_int_range


class CustomRandomForestRegressor():
    """
    A customized wrapper for the Random Forest Regressor
    """
    def __init__(self, df, y_name, random_state=False,
                test_size=0.1):
        """
        Custom wrapper for creating a forest regressor.
        The input is a dataframe. The y_name is the name of
        the column that is the predictor value (to be predicted).

        The usecase is, given a dataframe, call this function.
        It generates the tree and creates an API to call the values

        obj = CustomRandomForestRegressor(df, "y")
        obj.predict(new_x)

        Input parameters:
            - df Pandas Dataframe for the input
            - y_name   : String, name of the predictor value
            - random_state :  Boolean to either set a non-random
                              state (False) which will result
                              in a static random state
                              If true, then it will be random
            - test_size  : Size of split in train/test split
        """
        if random_state:
            rand_state = rand_int_range()
        else:
            rand_state = 199

        x_names = verify_name_in_series(df, y_name)

        X = df[x_names]
        Y = df[y_name]

        self.model = RandomForestRegressor(n_estimators=len(x_names),
                                           random_state=rand_state)
        self.x_train, self.x_test, self.y_train, self.y_test = \
                    train_test_split(X, Y, test_size=test_size)
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x):
        """Predict the values in x from the model"""
        return self.model.predict(x)

    def r2(self):
        """Return the r2_score for the regressor"""
        return r2_score(self.predict(self.x_test), self.y_test)


def profile_correlation(df, no_iters=100):
    """
    Profile the correlation between samples by running a CustomRandomForest
    Regressor several times on each variable.
    The resulting plot shows the random distribtuion between using the other
    variables to predict the noted variable. E.g. if the legend says pH then
    all the other variables are used to predict it using a random forest
    regressir. The system is run no_iters times and each time has a random
    seed. This allows you to estimate the probability 
    """
    variables = df.columns
    df_dict = {}
    for y_name in tqdm(list(variables)):
        temp_store = np.zeros((no_iters, ))
        for i in range(no_iters):
            obj = CustomRandomForestRegressor(df, y_name, random_state=True)
            temp_store[i] = obj.r2()
        df_dict[y_name] = sorted(temp_store)
    x = (np.arange(no_iters)+1)/no_iters
    df = pd.DataFrame(df_dict)
    df["percentage"] = x
    ax = df.plot(x="percentage", figsize=(12, 10))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return ax, df


