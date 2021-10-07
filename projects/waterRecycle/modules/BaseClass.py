"""BaseClass.py
A base class that implements some interesting
features allowing for the usecase of inheretance
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# PCA
from sklearn.decomposition import PCA


class BaseClass():
    @staticmethod
    def plot(df):
        """A generic function call to plot a dataframe"""
        colnames = df.columns
        fig, ax = plt.subplots(figsize=(12, 10))
        [n, m] = df.shape
        x = range(n)
        for i in range(m):
            ax.plot(x, df[colnames[i]], label=colnames[i])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=3, fancybox=True, shadow=True)

        return fig, ax

    @staticmethod
    def corr_heatmap(df, *args, **kwargs):
        """Compute the correlation between columns in dataframe
        and plot as a heatmap
        Keywords are passed forwards into the pandas.DataFrame.corr
        function
        """
        if not kwargs:
            return sns.heatmap(df.corr(),
                               xticklabels=df.columns,
                               yticklabels=df.columns)
        else:
            return sns.heatmap(df.corr(kwargs),
                               xticklabels=df.columns,
                               yticklabels=df.columns)

    @staticmethod
    def pairing(df, *args, **kwargs):
        """Plot a PairGrid object for each of the series against the 
        other"""
        g = sns.PairGrid(df)
        g.map(sns.scatterplot)

    @staticmethod
    def distributions(df, *args, **kwargs):
        """Plot the distribution of each columns using the
        seaborn library.
        Returns fig, ax (figure and axis handles)
        """
        [n, m] = df.shape
        names = df.columns
        fig, ax = plt.subplots(m, 1, figsize=(12, 10))
        for i in range(m):
            signal =  df[names[i]]
            sns.histplot(signal, ax=ax[i])

        fig.tight_layout()

        return fig, ax

    @staticmethod
    def _pca(df_n, *args, **kwargs):
        """Do a PCA of the dataaset and select the 3d values
        with the highest information stage, the user must
        make sure to 'NORMALIZE' features before use.
        Returns:
            pca_data : sklearn object of type PCA with infomration
            df_out   : pandas dataframe with reduced values
            """
        pca_obj = PCA(n_components=3)
        pca_data = pca_obj.fit_transform(df_n)
        df_out = pd.DataFrame(data=pca_data)
        return pca_data, df_out
