import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Dropout, Activation
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import glob
import joblib
from sklearn.utils import shuffle

def swat_create_a4_a5():
    import pandas as pd
    df_normal = pd.read_csv('../data/swat/0_normal.csv')
    df_attack = pd.read_csv('../data/swat/1_attack.csv')

    for column in df_normal:
        df_normal.loc[df_normal[column] == 'Inactive', column] = 0
        df_normal.loc[df_normal[column] == 'Active', column] = 1

    for column in df_attack:
        df_attack.loc[df_attack[column] == 'Inactive', column] = 0
        df_attack.loc[df_attack[column] == 'Active', column] = 1

    df_attack['label'] = 1
    df_normal['label'] = 0

    df = df_normal.append(df_attack, ignore_index=False)
    labels = df.label
    del df['label']
    del df['GMT +0']

    print(df)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, shuffle=True)

    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

    joblib.dump(data, "../data/swat/a45")

def swat_create_dataset():
    """Creates the SWaT dataset base on the two excel files.
    """
    frames = []
    label_enc = None

    for f in ['data/swat/attack.xlsx', 'data/swat/normal_v1.xlsx']:
        df = pd.read_excel(f)

        df.columns = df.iloc[[0]].values[0]
        df = df.iloc[1:]
        df = df.iloc[53:].copy()

        df = df.rename(columns=lambda x: x.replace(" ", ""))
        df['Normal/Attack'] = df['Normal/Attack'].map(lambda x: x.replace(" ", ""))

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by=['Timestamp'], ascending=True)

        if not label_enc:
            label_enc = preprocessing.LabelEncoder()
            df['Normal/Attack'] = label_enc.fit_transform(df['Normal/Attack'].values)
        else:
            df['Normal/Attack'] = label_enc.transform(df['Normal/Attack'].values)

        frames.append(df)

    y_attack = frames[0]['Normal/Attack'].values
    y_normal = frames[1]['Normal/Attack'].values

    del frames[0]['Normal/Attack']
    del frames[1]['Normal/Attack']

    X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(frames[0], y_attack, test_size=0.25,
                                                                                    shuffle=False)
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(frames[0], y_attack, test_size=0.25,
                                                                                    shuffle=False)

    data = {
        'X_attack_train': X_attack_train,
        'X_attack_test': X_attack_test,
        'X_normal_train': X_normal_train,
        'X_normal_test': X_normal_test,
        'y_attack_train': y_attack_train,
        'y_attack_test': y_attack_test,
        'y_normal_train': y_normal_train,
        'y_normal_test': y_normal_test
    }

    joblib.dump(data, "data/swat/dataset")


def swat_get_all_shuffled(data):
    """Gets the train and test SWaT dataset, but
       it is shuffled before.

    Args:
        data ([type]): The raw SWaT dataset.

    Returns:
        [type]: The shuffled data.
    """
    X_train, y_train, X_test, y_test = swat_get_all(data)
    return shuffle(X_train, y_train), shuffle(X_test, y_test)


def swat_get_all(data):
    """Concatenates the created SWaT dataset so that
       a test and train set exist.

    Args:
        data ([type]): The raw SWaT dataset.

    Returns:
        [type]: A train and test set.
    """
    X_train = pd.concat([data['X_normal_train'], data['X_attack_train']])
    y_train = np.concatenate((data['y_normal_train'], data['y_attack_train']))

    X_test = pd.concat([data['X_normal_test'], data['X_attack_test']])
    y_test = np.concatenate((data['y_normal_test'], data['y_attack_test']))

    return X_train, y_train, X_test, y_test


def swat_load_data():
    import joblib
    return joblib.load("../data/swat/dataset")


if __name__ == '__main__':
    #swat_create_dataset()
    swat_create_a4_a5()
