from matplotlib import pyplot as plt
from matplotlib.cm import PuOr
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from pycebox.ice import ice, ice_plot
import joblib
from sklearn import preprocessing


def run_ice(df, y, model, feature):
    """Calculates and plots an ICE curve for the feature of a given model.

    Args:
        df (DataFrame): The samples which are used to calculate the ICE.
        y (list): The labels for the samples.
        model ([type]): The model which predict function is used.
        feature ([type]): The column name of the feature which is analyzed.
    """
    print("Running ICE for feature", feature)
    ice_df = ice(df, feature, model.predict, num_grid_points=10)

    fig, (data_ax, ice_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 6))
    data_ax.scatter(df[feature], y, c='k', alpha=0.5)

    maxrow = df.loc[df[feature].idxmax()]

    data_ax.set_xlim(-0.1, maxrow[feature] * 1.1)
    data_ax.set_xlabel(feature)
    data_ax.set_ylabel('$y$')
    data_ax.set_title("Data: " + str(type(model)))

    ice_plot(ice_df, frac_to_plot=1,
             # color_by='crc rate', cmap=plt.get_cmap('cividis') ,
             alpha=0.25,
             c='k',
             ax=ice_ax)

    ice_ax.set_xlabel(feature)
    ice_ax.set_ylabel('$y$')
    ice_ax.set_title('ICE curves')


def get_model_data(model):
    df = model['data']['df'].sample(500)
    y = df['binary result'].values
    del df['binary result']

    return df, y


def rf_model():
    model = joblib.load("results/models/rf-full")
    df, y = get_model_data(model)

    return model, df, y


def svm_model():
    model = joblib.load("results/models/svm-full")
    df, y = get_model_data(model)

    return model, df, y


def svm_linear_model():
    model = joblib.load("results/models/svm-linear-full")
    df, y = get_model_data(model)

    return model, df, y


def nn_model():
    model = joblib.load("results/models/nn-full")
    df, y = get_model_data(model)

    min_max_scaler = preprocessing.MinMaxScaler()
    df[df.columns] = min_max_scaler.fit_transform(df[df.columns])

    return model, df, y


def nn_linear_model():
    model = joblib.load("results/models/nn-linear-full")
    df, y = get_model_data(model)

    min_max_scaler = preprocessing.MinMaxScaler()
    df[df.columns] = min_max_scaler.fit_transform(df[df.columns])

    return model, df, y


if __name__ == '__main__':
    """Calculates an ICE for each feature of a model and plots the curves.
    """
    nnl_model, nnl_df, nnl_y = rf_model()

    for column in nnl_df.columns:
        run_ice(nnl_df, nnl_y, nnl_model['box'], column)

    plt.show()
