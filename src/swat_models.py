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

# In this file the models based on the SWaT dataset are created.

# This are the different stages P1 - P6 and their corresponding features
p1 = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102']
p2 = ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206']
p3 = ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302']
p4 = ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401']
p5 = ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
      'PIT502', 'PIT503']
p6 = ['FIT601', 'P601', 'P602', 'P603']


def swat_train_svm_linear_pX(X_train, y_train, X_test, y_test, subset):
    gamma = 0.0008181483058667633

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    scaler = preprocessing.StandardScaler()
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    X_test = X_test[subset]
    X_train = X_train[subset]

    # X_train, y_train = _preprocess_svm(X_train, y_train, 4)
    # X_test, y_test = _preprocess_svm(X_test, y_test, 4)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    print(X_train.shape)

    from sklearn.svm import LinearSVC
    model = LinearSVC(max_iter=4000)
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return model


def swat_train_svm_pX(X_train, y_train, X_test, y_test, subset):
    gamma = 0.0008181483058667633

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    scaler = preprocessing.StandardScaler()
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    X_test = X_test[subset]
    X_train = X_train[subset]

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    print(X_train.shape)

    model = svm.SVC(gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return model


def swat_train_nn(X_train, y_train, X_test, y_test):
    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    scaler = preprocessing.StandardScaler()
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    model = Sequential()
    model.add(Dense(300, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train.to_numpy(), y_train, epochs=5, batch_size=15)

    y_true, y_pred = y_test, model.predict(X_test.to_numpy())
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'nn',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }


def swat_train_linear_nn(X_train, y_train, X_test, y_test):
    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    scaler = preprocessing.StandardScaler()
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    model = Sequential()
    model.add(Dense(300, ))
    model.add(Activation('linear'))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Activation('linear'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train.to_numpy(), y_train, epochs=5, batch_size=15)

    y_true, y_pred = y_test, model.predict(X_test.to_numpy())
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'linear_nn',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }


def swat_train_bayes(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'bayes',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }


def swat_train_logistic_regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    window_size = 4
    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    # X_train, y_train = _preprocess_svm(X_train, y_train, window_size)
    # X_test, y_test = _preprocess_svm(X_test, y_test, window_size)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = LogisticRegression(verbose=1)
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))
    return {
        'data': data,
        'box': model,
        'type': 'lr',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }


def swat_train_rf(X_train, y_train, X_test, y_test):
    print(X_train.head())

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(X_train, y_train)

    y_true, y_pred = y_test, [int(round(x)) for x in rf.predict(X_test)]
    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': rf,
        'type': 'rf',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }



def _preprocess_svm(X, y, window_size):
    X = X.to_numpy()

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    result = []
    result_y = []

    for idx in range(0, len(X) - window_size - 1):
        steps = X[idx:idx + window_size]
        labels = y[idx:idx + window_size]
        any_zero = any(x == 0 for x in labels)

        l = 1
        if any_zero:
            l = 0

        flat_list = []
        for sublist in steps:
            flat_list.extend(sublist)

        result.append(flat_list)
        result_y.append(l)

    return np.array(result), np.array(result_y)


def _preprocess_lstm(X, y, timesteps):
    result = []
    result_y = []

    for idx in range(0, len(X) - timesteps - 1):
        steps = X[idx:idx + timesteps]
        labels = y[idx:idx + timesteps]
        any_zero = any(x == 0 for x in labels)

        l = 1
        if any_zero:
            l = 0

        result.append(steps)
        result_y.append(l)

    return np.array(result), np.array(result_y)


def swat_train_lstm(X_train, y_train, X_test, y_test):
    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    time_steps = 14
    feature_size = 51

    X_train, y_train = _preprocess_lstm(X_train, y_train, time_steps)
    X_test, y_test = _preprocess_lstm(X_test, y_test, time_steps)

    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(LSTM(32, input_shape=(time_steps, feature_size), dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(X_train, y_train, epochs=5, batch_size=12, verbose=1, shuffle=True)

    y_true, y_pred = y_test, model.predict(X_test)
    np.place(y_pred, y_pred, np.round(y_pred))

    print(classification_report(y_true, y_pred))
    return model


def swat_train_svm(X_train, y_train, X_test, y_test):
    window_size = 4
    gamma = 0.0008181483058667633

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    X_train, y_train = _preprocess_svm(X_train, y_train, window_size)
    X_test, y_test = _preprocess_svm(X_test, y_test, window_size)

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    model = svm.SVC(gamma=gamma, kernel='rbf', C=100)
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'svm',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }



def swat_train_svm_single(X_train, y_train, X_test, y_test):
    gamma = 0.0008181483058667633

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'svm',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }


def swat_train_svm_linear_single(X_train, y_train, X_test, y_test):
    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    model = svm.LinearSVC(max_iter=4000)
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'svm',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }


def swat_train_svm_linear(X_train, y_train, X_test, y_test):
    window_size = 4
    gamma = 0.0008181483058667633

    if 'Timestamp' in X_train:
        del X_train['Timestamp']
    if 'Timestamp' in X_test:
        del X_test['Timestamp']

    X_train, y_train = _preprocess_svm(X_train, y_train, window_size)
    X_test, y_test = _preprocess_svm(X_test, y_test, window_size)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test, shuffle(X_test, y_test)

    print(X_train.shape)

    model = svm.LinearSVC(verbose=True, max_iter=4000)
    model.fit(X_train, y_train)

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    return {
        'data': data,
        'box': model,
        'type': 'svm_linear',
        'metrics': classification_report(y_true, y_pred, output_dict=True)
    }



def swat_rf_model():
    import joblib
    return joblib.load("../results/models/swat/rf_new")


def swat_svm_model():
    import joblib
    return joblib.load("../results/models/swat/svm")


def swat_lr_model():
    import joblib
    return joblib.load("../results/models/swat/lr")


def swat_nn_model():
    import joblib
    return joblib.load("../results/models/swat/nn")

def swat_nn_linear_model():
    import joblib
    return joblib.load("../results/models/swat/nn-linear")


def swat_svm_single_model():
    import joblib
    return joblib.load("../results/models/swat/svm-single")


def swat_svm_linear_model():
    import joblib
    return joblib.load("../results/models/swat/svm-linear")


def swat_svm_linear_single_model():
    import joblib
    return joblib.load("../results/models/swat/svm-linear-single")


def swat_lstm_model():
    import joblib
    return joblib.load("../results/models/swat/lstm")


def swat_bayes_model():
    import joblib
    return joblib.load("../results/models/swat/bayes")


def swat_svm_p3():
    import joblib
    return joblib.load("../results/models/swat/svm-p3")


def swat_svm_linear_p3():
    import joblib
    return joblib.load("../results/models/swat/svm-linear-p3")


def print_results_svm_px():
    count = 1
    for entry in [p1, p2, p3, p4, p5, p6]:
        data = swat_load_data()
        X_train, y_train, X_test, y_test = swat_get_all(data)
        svm = joblib.load("../results/models/swat/svm-p" + str(count))
        X_test = X_test[entry]
        scaler = preprocessing.StandardScaler()
        X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
        y_true, y_pred = y_test, svm.predict(X_test)
        print(classification_report(y_true, y_pred))

        count = count + 1


if __name__ == '__main__':
    from swat_data import *

    data = swat_load_data()
    X_train, y_train, X_test, y_test = swat_get_all(data)

    svm = swat_train_svm_single(X_train, y_train, X_test, y_test)
    joblib.dump(svm, "../results/models/swat/svm")

    # data = swat_load_data()
    # X_train, y_train, X_test, y_test = swat_get_all(data)

    #    svm = swat_train_svm(X_train, y_train, X_test, y_test)
    #    joblib.dump(svm, "results/models/swat/svm-timesteps")

    # raise Exception()

    #data = swat_load_data()
    #X_train, y_train, X_test, y_test = swat_get_all(data)

    #nn_lin = swat_train_linear_nn(X_train, y_train, X_test, y_test)
    #joblib.dump(nn_lin, "../results/models/swat/nn-linear")

    #nn = swat_train_nn(X_train, y_train, X_test, y_test)
    #joblib.dump(nn, "../results/models/swat/nn")

    # raise Exception()

    # rf = swat_train_rf(X_train, y_train, X_test, y_test)
    # joblib.dump(rf, "results/models/swat/rf-new")

    # svm = joblib.load("results/models/swat/rf")
    # del X_test['Timestamp']

    # scaler = preprocessing.StandardScaler()
    # X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])

    # X_test, y_test = _preprocess_lstm(X_test, y_test, 14)
    # print("predictiong...")
    # y_true, y_pred = y_test, [int(round(x)) for x in svm.predict(X_test)]
    # print("finish")
    # print(classification_report(y_true, y_pred))

    # count = 1
    # for entry in [p1, p2, p3, p4, p5, p6]:
    #   data = swat_load_data()
    #   X_train, y_train, X_test, y_test = swat_get_all(data)
    #   svm = swat_train_svm_linear_pX(X_train, y_train, X_test, y_test, entry)
    #   joblib.dump(svm, "results/models/swat/svm-linear-p" + str(count))
    #   count = count + 1

    # count = 1
    # for entry in [p1, p2, p3, p4, p5, p6]:
    #    data = swat_load_data()
    #    X_train, y_train, X_test, y_test = swat_get_all(data)
    #    svm = swat_train_svm_pX(X_train, y_train, X_test, y_test, entry)
    #    joblib.dump(svm, "results/models/swat/svm-p" + str(count))
    #    count = count + 1

    # raise Exception()

    data = swat_load_data()
    X_train, y_train, X_test, y_test = swat_get_all(data)

    svm = swat_train_svm_linear_single(X_train, y_train, X_test, y_test)
    joblib.dump(svm, "../results/models/swat/svm-linear")

    # data = swat_load_data()
    # X_train, y_train, X_test, y_test = swat_get_all(data)

    # bayes = swat_train_svm_single(X_train, y_train, X_test, y_test)
    # joblib.dump(bayes, "results/models/swat/svm-single")

    #data = swat_load_data()
    #X_train, y_train, X_test, y_test = swat_get_all(data)

    #bayes = swat_train_bayes(X_train, y_train, X_test, y_test)
    #joblib.dump(bayes, "../results/models/swat/bayes")

    #lr = swat_train_logistic_regression(X_train, y_train, X_test, y_test)
    #joblib.dump(lr, "../results/models/swat/lr")

    # data = swat_load_data()
    # X_train, y_train, X_test, y_test = swat_get_all(data)

    # bayes1 = swat_train_bayes(X_train, y_train, X_test, y_test)
    # joblib.dump(bayes1, "results/models/swat/bayes1")

    #data = swat_load_data()
    #X_train, y_train, X_test, y_test = swat_get_all(data)

    #svm_linear = swat_train_svm_linear(X_train, y_train, X_test, y_test)
    #joblib.dump(svm_linear, "../results/models/swat/svm-linear")

    #data = swat_load_data()
    #X_train, y_train, X_test, y_test = swat_get_all(data)

    #svm_linear1 = swat_train_svm_linear(X_train, y_train, X_test, y_test)
    #joblib.dump(svm_linear1, "results/models/swat/svm-linear1")

    # data = swat_load_data()
    # X_train, y_train, X_test, y_test = swat_get_all(data)

    # lstm = swat_train_lstm(X_train, y_train, X_test, y_test)
    # joblib.dump(lstm, "results/models/swat/lstm")

    # data = swat_load_data()
    # X_train, y_train, X_test, y_test = swat_get_all(data)

    # lstm1 = swat_train_lstm(X_train, y_train, X_test, y_test)
    # joblib.dump(lstm1, "results/models/swat/lstm1")

    #data = swat_load_data()
    #X_train, y_train, X_test, y_test = swat_get_all(data)

    #svm1 = swat_train_svm(X_train, y_train, X_test, y_test)
    #joblib.dump(svm1, "../results/models/swat/svm1")

    #data = swat_load_data()
    #X_train, y_train, X_test, y_test = swat_get_all(data)

    #svm = swat_train_svm(X_train, y_train, X_test, y_test)
    #joblib.dump(svm, "../results/models/swat/svm")

    #data = swat_load_data()
    #train, test = swat_get_all_shuffled(data)

    #rf = swat_train_rf(X_train, y_train, X_test, y_test)
    #joblib.dump(rf, "../results/models/swat/rf_new")

    # data = swat_load_data()
    # train, test = swat_get_all_shuffled(data)

    # rf1 = swat_train_rf(train[0], train[1], test[0], test[1])
    # joblib.dump(rf1, "results/models/swat/rf1")
