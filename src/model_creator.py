from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Dropout, Activation
from sklearn import preprocessing
from sklearn import svm
import joblib
from gp_rule_model import GasPipelineRuleModel


class ModelCreator():
    """Creates and trains different models on the gas pipeline dataset.
    """

    def train_rule_model(self, data):
        X_train = data['X_train'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining rule model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        blackbox = GasPipelineRuleModel()
        blackbox.fit(X, y)

        X_test = data['X_test']
        y_test = data['y_test']
        y_true, y_pred = y_test, blackbox.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': blackbox,
            'type': 'rf',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_linear_regression(self, data):
        from sklearn.linear_model import LogisticRegression

        X_train = data['X_train_norm'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining logistic regession model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        model = LogisticRegression()
        model.fit(X, y)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, model.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': model,
            'type': 'logistic-regression',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_svm_non_linear(self, data):
        X_train = (data['X_train_norm']).to_numpy()
        y_train = data['y_train']

        print("\nTraining non-linear SVM model ...")
        print("X:", X_train.shape)
        print("Y:", y_train.shape)

        model = svm.SVC(kernel='rbf', gamma=0.2689, C=107.411)
        model.fit(X_train, y_train)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, model.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': model,
            'type': 'svm',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_svm(self, data):
        X_train = (data['X_train_norm']).to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining linear SVM model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        model = svm.SVC(kernel='linear')
        model.fit(X_train, y_train)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, model.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': model,
            'type': 'svm-linear',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_nn(self, data):
        X_train = data['X_train_norm'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining NN model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        model = Sequential()
        model.add(Dense(500, ))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(300))
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=12, batch_size=15)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, model.predict(X_test)
        np.place(y_pred, y_pred, np.round(y_pred))

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': model,
            'type': 'nn',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_linear_nn(self, data):
        X_train = data['X_train_norm'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining linear NN model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        model = Sequential()
        model.add(Dense(500, ))
        model.add(Activation('linear'))
        model.add(Dropout(0.4))
        model.add(Dense(300))
        model.add(Activation('linear'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=12, batch_size=15)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, model.predict(X_test)
        np.place(y_pred, y_pred, np.round(y_pred))

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': model,
            'type': 'nn-linear',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_bayes(self, data):
        X_train = data['X_train'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining bayes model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        blackbox = GaussianNB()
        blackbox.fit(X_train, y_train)

        X_test = data['X_test']
        y_test = data['y_test']
        y_true, y_pred = y_test, blackbox.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': blackbox,
            'type': 'rf',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_bayes_norm(self, data):
        X_train = data['X_train_norm'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining bayes model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        blackbox = GaussianNB()
        blackbox.fit(X_train, y_train)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, blackbox.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': blackbox,
            'type': 'rf',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_rf(self, data):
        X_train = data['X_train'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining RF model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        blackbox = RandomForestClassifier(n_estimators=20)
        blackbox.fit(X, y)

        X_test = data['X_test']
        y_test = data['y_test']
        y_true, y_pred = y_test, blackbox.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': blackbox,
            'type': 'rf',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }

    def train_rf_norm(self, data):
        X_train = data['X_train_norm'].to_numpy()
        y_train = data['y_train']
        X, y = shuffle(X_train, y_train)

        print("\nTraining RF model ...")
        print("X:", X.shape)
        print("Y:", y.shape)

        blackbox = RandomForestClassifier(n_estimators=20)
        blackbox.fit(X, y)

        X_test = data['X_test_norm']
        y_test = data['y_test']
        y_true, y_pred = y_test, blackbox.predict(X_test)

        print(classification_report(y_true, y_pred))

        return {
            'data': data,
            'box': blackbox,
            'type': 'rf',
            'metrics': classification_report(y_true, y_pred, output_dict=True)
        }


nn_path = "../results/models/nn-full"
nn_linear_path = "../results/models/nn-linear-full"
svm_path = "../results/models/svm-full"
svm_linear_path = "../results/models/svm-full-linear"
rf_path = "../results/models/rf-full"
logistic_regression_path = "../results/models/logistic-regression"
bayes_path = "../results/models/bayes"
rulemodel_path = "../results/models/custom-rules"


def load_rf():
    import joblib
    return joblib.load(rf_path)


def load_svm():
    import joblib
    return joblib.load(svm_path)


def load_svm_linear():
    import joblib
    return joblib.load(svm_linear_path)


def load_nn():
    import joblib
    return joblib.load(nn_path)


def load_nn_linear():
    import joblib
    return joblib.load(nn_linear_path)


def load_rf_duplicate():
    import joblib
    return joblib.load(rf_path + "-1")


def load_svm_duplicate():
    import joblib
    return joblib.load(svm_path + "-1")


def load_svm_linear_duplicate():
    import joblib
    return joblib.load(svm_linear_path + "-1")


def load_nn_duplicate():
    import joblib
    return joblib.load(nn_path + "-1")


def load_nn_linear_duplicate():
    import joblib
    return joblib.load(nn_linear_path + "-1")


def load_logistic_regression():
    import joblib
    return joblib.load(logistic_regression_path)


def load_logistic_regression_duplicate():
    import joblib
    return joblib.load(logistic_regression_path + "-1")


def load_bayes():
    import joblib
    return joblib.load(bayes_path)


def load_rule_model():
    import joblib
    return joblib.load(rulemodel_path)


if __name__ == '__main__':
    """Trains all model types on the gas pipeline dataset
       and persists the results on disk.
    """
    from dataset import *
    import joblib

    creator = ModelCreator()

    #df = load_traffic_data()
    # model = creator.train_rule_model(df)
    # joblib.dump(model, rulemodel_path, compress=9)

    #df = load_traffic_data()
    #model = creator.train_bayes(df)
    #joblib.dump(model, bayes_path, compress=9)

    #df = load_traffic_data()
    #model = creator.train_linear_regression(df)
    #joblib.dump(model, logistic_regression_path, compress=9)

    # df = load_full_data()
    # model = creator.train_linear_regression(df)
    # joblib.dump(model, logistic_regression_path + "-1", compress=9)

    #df = load_traffic_data()
    #model = creator.train_svm(df)
    #joblib.dump(model, svm_linear_path, compress=9)

    # df = load_full_data()
    # model = creator.train_svm(df)
    # joblib.dump(model, svm_linear_path + "-1", compress=9)

    #df = load_traffic_data()
    #model = creator.train_svm_non_linear(df)
    #joblib.dump(model, svm_path, compress=9)

    # df = load_full_data()
    # model = creator.train_svm_non_linear(df)
    # joblib.dump(model, svm_path + "-1", compress=9)

    #df = load_traffic_data()
    #model = creator.train_nn(df)
    #joblib.dump(model, nn_path, compress=9)

    # df = load_full_data()
    # model = creator.train_nn(df)
    # joblib.dump(model, nn_path + "-1", compress=9)

    #df = load_traffic_data()
    #model = creator.train_linear_nn(df)
    #joblib.dump(model, nn_linear_path, compress=9)

    # df = load_full_data()
    # model = creator.train_linear_nn(df)
    # joblib.dump(model, nn_linear_path + "-1", compress=9)

    #df = load_traffic_data()
    #model = creator.train_rf(df)
    #joblib.dump(model, rf_path, compress=9)

    # df = load_full_data()
    # model = creator.train_rf(df)
    # joblib.dump(model, rf_path + "-1", compress=9)


    # --- calculating surrogates
    #number_of_surrogates = 5
    number_of_surrogates = 1
    #df = load_protocol_data()
    df = load_full_data()
    models = [
        {'method': creator.train_bayes_norm, 'path': 'bayes'},
        #{'method': creator.train_linear_regression, 'path': 'lr'},
        #{'method': creator.train_nn, 'path': 'nn'},
        #{'method': creator.train_linear_nn, 'path': 'nn-linear'},
        {'method': creator.train_rf_norm, 'path': 'rf'},
        #{'method': creator.train_svm, 'path': 'svm-linear'},
        #{'method': creator.train_svm_non_linear, 'path': 'svm'},
    ]

    for model in models:
        print(f"Starting with {model['path']}")
        for i in range(number_of_surrogates):
            print(f"Now training {model['path']} in round {i}")
            trained_model = model['method'](df)
            joblib.dump(trained_model, "../results/models/" + model['path'] + f"-norm", compress=9)
            #joblib.dump(trained_model, "../results/models/gas-pipeline/protocol-data/" + model['path'], compress=9)


