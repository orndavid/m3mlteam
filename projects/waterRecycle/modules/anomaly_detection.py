"""anomaly_detection.py
Create a system that takes a time series in and
builds an anomly detection around the singals and
highlights signals that are outside the expected
distribution
"""
from modules.BaseClass import BaseClass
from modules.utils import normalize
from modules.utils import _cnorm


class AD(BaseClass):
    """Anomaly detection class object. A wrapper that takes
    in a time series dataframe and opens up an API to do
    time series analysis"""
    def __init__(self, df):
        self.df = df

    def linear_plot(self):
        """Plot all the series as a normalized line on a single
        plot to get the high level overview"""
        return AD.plot(self.df)

    def linear_plot_norm(self):
        """Plot the series as a normalized dataset
            return fig, ax
        """
        return AD.plot(self.df.apply(_cnorm))

    def corr(self, *args, **kwargs):
        """Create a figure of the correlation between the
        columns"""
        return AD.corr_heatmap(self.df, *args, **kwargs)

    def stats(self):
        """Print the dataframe statistics on screen
            return fig, ax
        """
        self.df.describe(include="all")

    def dists(self, *args, **kwargs):
        """Plot the distributions for each signal"""
        AD.distributions(self.df, *args, **kwargs)

    def pca(self, *args, **kwargs):
        """Run a PCA and reduce down to 3 dimensions"""
        if "method" not in kwargs:
            method = "range"
        else:
            method = kwargs["method"]
        df = normalize(method, self.df)
        return AD._pca(df, *args, **kwargs)





