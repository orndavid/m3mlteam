"""nn.py
Development of a neural network framework
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.utils import verify_name_in_series
from modules.utils import rand_int_range

class NN():
    """
    Generic wrapper for a neural network build using pandas
    dataframe
    How to use:
        Give the series a dataframe with only numerical values (no
        categories or objects) and a y_name. The y_name determines
        which column to use for predictor value

    For specific inputs to control model see sklearn documentation
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    """
    def __init__(self, df, y_name, test_size=0.3, rand_state=True,
             *args, **kwargs):
        x_names = verify_name_in_series(df, y_name)
        X = df[x_names]
        Y = df[y_name]
        
        if rand_state:
            random_value = rand_int_range()
        else:
            random_value = 999

        self.x_train, self.x_test, self.y_train, self.y_test = \
                                train_test_split(X, Y, 
                                        test_size=test_size, 
                                        random_state=random_value)
        self.model = MLPRegressor(*args, **kwargs)

        self.model.fit(self.x_train, self.y_train)

    def predict(self, x):
        """Predict based on a specific input"""
        return self.model.predict(x)

    def r2(self):
        return r2_score(self.predict(self.x_test), self.y_test)



def profile_correlation(df, no_iters=100):
    """
    Profile the correlation between samples by running a NN 
    several times on each variable.
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
            obj = NN(df, y_name, test_size=0.1, random_state=True)
            temp_store[i] = obj.r2()
        df_dict[y_name] = sorted(temp_store)
    x = (np.arange(no_iters)+1)/no_iters
    df = pd.DataFrame(df_dict)
    df["percentage"] = x
    ax = df.plot(x="percentage", figsize=(12, 10))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return ax, df

