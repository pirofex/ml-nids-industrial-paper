import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Dropout, Activation
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers.recurrent import LSTM

def swat_train_nn(X_train, y_train, X_test, y_test):
    scaler = preprocessing.StandardScaler()
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    model = Sequential()
    model.add(Dense(400, ))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=5, batch_size=7)

    y_true, y_pred = y_test, model.predict(X_test.to_numpy())
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))
    return model


def swat_train_linear_nn(X_train, y_train, X_test, y_test):
    scaler = preprocessing.StandardScaler()
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    model = Sequential()
    model.add(Dense(400, ))
    model.add(Activation('linear'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('linear'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=5, batch_size=7)

    y_true, y_pred = y_test, model.predict(X_test.to_numpy())
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))
    return model


def swat_train_bayes(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    rf = GaussianNB()
    rf.fit(X_train, y_train)

    y_true, y_pred = y_test, rf.predict(X_test)
    print(classification_report(y_true, y_pred))

    return rf


def swat_train_logistic_regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = LogisticRegression(verbose=1)
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))
    return model


def swat_train_rf(X_train, y_train, X_test, y_test):
    print(X_train.head())

    rf = RandomForestRegressor(n_estimators=150, verbose=1, n_jobs=4)
    rf.fit(X_train, y_train)

    y_true, y_pred = y_test, [int(round(x)) for x in rf.predict(X_test)]
    print(classification_report(y_true, y_pred))

    return rf

def swat_train_svm_single(X_train, y_train, X_test, y_test):
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return model

def swat_train_svm_linear_single(X_train, y_train, X_test, y_test):
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = svm.LinearSVC()
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return model

if __name__ == '__main__':
    from swat_data import *

    a45data = joblib.load("data/swat/a45")
    bayes = swat_train_bayes(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(bayes, "results/a45_bayes")

    a45data = joblib.load("data/swat/a45")
    svm_linear = swat_train_svm_linear_single(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(svm_linear, "results/a45_svm_linear")

    a45data = joblib.load("data/swat/a45")
    svm = swat_train_svm_single(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(svm, "results/a45_svm")
    
    a45data = joblib.load("data/swat/a45")
    linear_nn = swat_train_linear_nn(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(linear_nn, "results/a45_nn_linear")

    a45data = joblib.load("data/swat/a45")
    nn = swat_train_nn(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(nn, "results/a45_nn")

    a45data = joblib.load("data/swat/a45")
    lr = swat_train_logistic_regression(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(lr, "results/a45_lr")

    a45data = joblib.load("data/swat/a45")
    rf = swat_train_rf(a45data['X_train'], a45data['y_train'], a45data['X_test'], a45data['y_test'])
    joblib.dump(rf, "results/a45_rf")

    # Plot feature importances of the random forest
    rf = joblib.load("results/a45_rf")
    a45data = joblib.load("data/swat/a45")
    features = a45data['X_train'].columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()