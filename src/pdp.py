import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


def plot_pdp(df, model, feature):
    """Calculates a one-dimensional PDP for the given feature.

    Args:
        df ([type]): The samples which are used for the calculation of the PDP.
        model ([type]): The model which predict function is used.
        feature ([type]): The feature column name.
    """
    try:
        del df['binary result']
    except KeyError:
        pass

    mypdp = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns, feature=feature)
    if len(mypdp.feature_grids) > 1:
        gradient = np.gradient(mypdp.pdp, mypdp.feature_grids)
    else:
        gradient = -1
    print(f"max gradient for {str(feature)}: {np.amax(gradient)}")
    fig, axes = pdp.pdp_plot(mypdp, feature, plot_lines=True, plot_pts_dist=True, center=True, show_percentile=True)


def plot_interaction_pdp(df, model, features):
    """Calculates a two-dimensional PDP for the given features.

    Args:
        df ([type]): The samples which are used for the calculation of the PDP.
        model ([type]): The model which predict function is used.
        feature ([type]): The features column names.
    """
    try:
        del df['binary result']
    except KeyError:
        pass

    inter1 = pdp.pdp_interact(model=model, dataset=df, model_features=df.columns, features=features)
    fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features, plot_type='contour',
                                      x_quantile=False, plot_pdp=True)


def _example_extended_grid():
    """Example where instead of building the value grid based on the provided
    samples a custom grid is used.
    """
    from model_creator import load_svm

    model = load_svm()
    box = model['box']
    df = model['data']['X_test_norm']
    df = df.sample(250)

    mypdp = pdp.pdp_isolate(model=box, dataset=df, model_features=df.columns, feature='deadband',
                            cust_grid_points=[-0.6, -0.5, -0.25, -0.1, 0, 0.00003, 0.0001, 0.001, 0.01, 0.02, 0.04, 0.1,
                                              0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9, 1.0])
    fig, axes = pdp.pdp_plot(mypdp, 'deadband', plot_lines=False, plot_pts_dist=True, center=True, show_percentile=True)
    plt.show()


def fibonacci(start, length, negative=False):
    if negative:
        if start > 0:
            print("You started a negative sequence with a positive number, this might give you unintended results.")
        f = [start, start - 1]
        for i in range(length - len(f)):
            if f[i + 1] < 0:
                f.append(f[i] + f[i + 1])
            else:
                f.append(f[i] - f[i + 1])
        f.reverse()
        return f
    else:
        if start < 0:
            print("You started a positive sequence with a negative number, this might give you unintended results.")
        f = [start, start + 1]
        for i in range(length - len(f)):
            f.append(f[i + 1] + f[i])
        return f


def calculate_pdp_fibonacci(df, model, feature):
    """Extends the value grid using fibonacci sampling and calculates PDP"""
    try:
        del df['binary result']
    except KeyError:
        pass
    values = df[feature]
    minimum_value = values.min()
    maximum_value = values.max()
    values = values.unique()
    values = values.tolist()
    length = 20
    if maximum_value > 0:
        values.extend(fibonacci(maximum_value, length))
    else:
        print(f"Max issue for {feature}: {maximum_value}")
        values.extend(fibonacci(0, length))
    if minimum_value < 0:
        values.extend(fibonacci(minimum_value, length, True))
    else:
        print(f"Min issue for {feature}: {minimum_value}")
        values.extend(fibonacci(-1, length, True))

    print(f"Used following grid: {values}")

    mypdp = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns, feature=feature, cust_grid_points=values)
    if len(mypdp.feature_grids) > 1:
        gradient = np.gradient(mypdp.pdp, mypdp.feature_grids)
    else:
        gradient = -1
    print(f"max gradient for {str(feature)}: {np.amax(gradient)}")
    #fig, axes = pdp.pdp_plot(mypdp, feature, plot_lines=True, plot_pts_dist=True, center=True, show_percentile=True)
    #plt.show()


if __name__ == '__main__':
    """Calculates PDP for all given features and plots them.

    Raises:
        Exception: [description]
        Exception: [description]
    """
    from model_creator import *
    from swat_data import *
    from dataset import *
    from swat_models import _preprocess_svm

    model_paths = [#"../results/models/gas-pipeline/protocol-data/lr",
                   #"../results/models/gas-pipeline/protocol-data/bayes",
                   #"../results/models/gas-pipeline/protocol-data/nn",
                   #"../results/models/gas-pipeline/protocol-data/nn-linear",
                   #"../results/models/gas-pipeline/protocol-data/rf",
                   #"../results/models/gas-pipeline/protocol-data/svm",
                   #"../results/models/gas-pipeline/protocol-data/svm-linear"
                    "../results/models/swat/rf"
                    ]
    for path in model_paths:
        print(f"now working with {path}")
        model = joblib.load(path)
        box = model['box']
        # gas pipeline
        #data = model['data']['X_test_norm']
        # swat
        data = model['data']['X_normal_test']
        # data = load_full_data()
        # data = data['X_test_norm']
        data = data.sample(500)
        if 'Timestamp' in data:
            del data['Timestamp']

        for entry in data.columns:
        #for entry in model['data']['df'].columns:
            if entry == 'system mode' or entry == 'binary result':
                continue
            plot_pdp(data, box, entry)
            #calculate_pdp_fibonacci(data,box,entry)

        plt.show()

    raise Exception()

    # This is an experiment with timesteps which are modelled by
    # a SVM. In this case the interaction of one feature (AIT201)
    # between a timestep is plotted.
    model = joblib.load("results/models/swat/svm-timesteps")

    data = swat_load_data()
    X_train, y_train, X_test, y_test = swat_get_all(data)

    del X_test['Timestamp']
    X_test, y_test = _preprocess_svm(X_test, y_test, 4)
    X_test, y_test = shuffle(X_test, y_test)

    samples = X_test[:50]
    df = pd.DataFrame(samples)

    scaler = preprocessing.StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    mypdp = pdp.pdp_interact(model=model, dataset=df, model_features=df.columns, features=[5, 56])
    fig, axes = pdp.pdp_interact_plot(mypdp, feature_names=['AIT201 (t-1)', 'AIT201 (t)'])
    plt.show()
    plot_interaction_pdp(data, box, ['length', 'crc rate'])

    plt.show()

    raise Exception()
