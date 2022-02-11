from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Dropout, Activation
from sklearn import preprocessing
from sklearn import svm
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# In this file the surrogate models are created.

def train_linear_svm_surrogate(X):
    y = X['binary result'].values.astype('int')
    del X['binary result']

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.svm import LinearSVC

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    return {
        'box': clf,
        'X': X,
        'y': y
    }


def train_svm_surrogate(X):
    y = X['binary result'].values.astype('int')
    del X['binary result']

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(
        SVC(), tuned_parameters
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    model = {
        'box': clf,
        'X': X,
        'y': y
    }

    return model


def train_rf_surrogate(X):
    y = X['binary result'].values
    del X['binary result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = [int(x) for x in np.linspace(start=20, stop=1000, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=20, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_params_
    print(best_params)

    max_depth = [None]

    if best_params['max_depth'] is not None:
        max_depth = np.linspace(start=best_params['max_depth'], stop=best_params['max_depth'] * 2, num=3,
                                dtype=np.int32)

    grid = {'n_estimators': np.linspace(start=best_params['n_estimators'], stop=best_params['n_estimators'] * 2, num=3,
                                        dtype=np.int32),
            'max_features': [best_params['max_features']],
            'max_depth': max_depth,
            'min_samples_split': np.linspace(start=best_params['min_samples_split'],
                                             stop=best_params['min_samples_split'] * 2, num=3, dtype=np.int32),
            'min_samples_leaf': np.linspace(start=best_params['min_samples_leaf'],
                                            stop=best_params['min_samples_leaf'] * 2, num=3, dtype=np.int32),
            'bootstrap': [best_params['bootstrap']]}

    grid_search = GridSearchCV(estimator=rf, param_grid=grid,
                               cv=3, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    print(best_params)
    best_grid = grid_search.best_estimator_

    try:
        predictions = [int(round(x)) for x in best_grid.predict(X_test)]
        y_test = y_test.tolist()

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        print('Accuracy:', accuracy)
        print('Recall:', recall)
        print('Precision:', precision)
        print('F1:', f1)
    except:
        pass

    model = {
        'box': best_grid,
        'X': X,
        'y': y
    }

    return model


def train_logistic_surrogate(X):
    from sklearn.linear_model import LogisticRegression

    y = X['binary result'].values
    del X['binary result']

    X_df = X.copy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_true, y_pred = y_test, model.predict(X_test)

    print(classification_report(y_true, y_pred))
    return model


def train_nn_surrogate(X):
    y = X['binary result'].values
    del X['binary result']

    X_df = X.copy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(100, ))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=10)

    return {
        'box': model,
        'X': X_df,
        'y': y
    }


def train_nn_linear_surrogate(X):
    y = X['binary result'].values
    del X['binary result']

    X_df = X.copy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(100, ))
    model.add(Activation('linear'))
    model.add(Dense(50))
    model.add(Activation('linear'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=10)

    out = {
        'box': model,
        'X': X_df,
        'y': y
    }

    y_true, y_pred = y_test, model.predict(X_test)
    np.place(y_pred, y_pred, np.round(y_pred))
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)

    print(classification_report(y_true, y_pred))
    return out


if __name__ == '__main__':
    """Here, the surrogate models for a logistic regression model are created.
    """
    from sklearn.neighbors import KernelDensity
    import numpy as np
    from model_creator import *
    from dataset import *
    import pandas as pd

    # actually not necessary for non-labelled step but due to implementation details it should be loaded in both cases
    model = joblib.load("../results/models/logistic-regression")

    # ---- no labelling ---
    X = model['data']['X_train_norm']
    X['binary result'] = np.array(model['data']['y_train'])

    # --- no labelling ---

    # ---- labelling ---
    # X = model['data']['X_test_norm']

    # Relabeling step
    # labels = []
    # for index, point in X.iterrows():
    #    inval = point.to_numpy()[:19].reshape(1, -1)
    #    label = model['box'].predict(inval)[0]
    #    labels.append(label)

    # labels = np.array(labels)
    # np.place(labels, labels, np.round(labels))
    # X['binary result'] = labels

    # --- labelling ---

    print(X.describe())

    #nn_linear = train_nn_linear_surrogate(X.copy())
    #joblib.dump(nn_linear, "../results/surrogates_nol/logistic/nn-linear", compress=9)

    #nn = train_nn_surrogate(X.copy())
    #joblib.dump(nn, "../results/surrogates_nol/nn", compress=9)

    #svm = train_svm_surrogate(X.copy())
    #joblib.dump(svm, "../results/surrogates_nol/svm", compress=9)

    #svm_linear = train_linear_svm_surrogate(X.copy())
    #joblib.dump(svm_linear, "../results/surrogates_nol/svm-linear", compress=9)

    #rf = train_rf_surrogate(X.copy())
    #joblib.dump(rf, "../results/surrogates_nol/rf", compress=9)

    logistic = train_logistic_surrogate(X.copy())
    joblib.dump(logistic, "../results/surrogates_nol/logistic", compress=9)
