def swat_a45_h_calculation():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from swat_data import *
    from swat_models import *
    from h_statistic import *
    from swat_models import _preprocess_svm, _preprocess_lstm

    import joblib
    import pandas as pd
    import itertools

    models = [
        {'model': joblib.load("results/a45_svm"), 'path': 'svm'},
        {'model': joblib.load("results/a45_svm_linear"), 'path': 'svm_linear'},
        {'model': joblib.load("results/a45_nn"), 'path': 'nn'},
        {'model': joblib.load("results/a45_nn_linear"), 'path': 'nn_linear'},
        {'model': joblib.load("results/a45_rf"), 'path': 'rf'},
        {'model': joblib.load("results/a45_lr"), 'path': 'lr'},
        {'model': joblib.load("results/a45_bayes"), 'path': 'bayes'},
    ]

    for model in models:
        for i in range(1):
            data = joblib.load("data/swat/a45")
            X_test = data['X_test']
            samples = X_test.sample(300)

            if model['path'] != 'rf':
                print('scaling ..')
                scaler = preprocessing.StandardScaler()
                samples[samples.columns] = scaler.fit_transform(samples[samples.columns])

            print("Calculating for", model['path'])
            vals = calculate_h_values(samples, model['model'], "results/h-vals/" + "a45_hvals-" + model['path'] + str(i))

if __name__ == '__main__':
    """Here the H-Statistic values for the SWaT models are calculated.
       Similar to the stuff happening in 'h_statistic.py' for the gas pipeline models.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from swat_data import *
    from swat_models import *
    from h_statistic import *
    from swat_models import _preprocess_svm, _preprocess_lstm

    import joblib
    import pandas as pd
    import itertools

    p1 = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102']
    p2 = ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206']
    p3 = ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302']
    p4 = ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401']
    p5 = ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501',
          'PIT502', 'PIT503']
    p6 = ['FIT601', 'P601', 'P602', 'P603']
    comblist = [p1, p2, p3, p4, p5, p6]

    svm_single_models = [
        # { 'model': joblib.load("results/models/swat/nn"), 'path': 'nn'},
        {'model': joblib.load("results/models/swat/svm-timesteps"), 'path': 'svm-timesteps3'},
    ]

    for model in svm_single_models:
        for i in range(1):
            data = swat_load_data()
            X_train, y_train, X_test, y_test = swat_get_all(data)
            del X_train['Timestamp']
            del X_test['Timestamp']

            # scaler = preprocessing.StandardScaler()
            # X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])

            # X_test = X_test.sample(100)
            # y_test = y_test[:100]

            X_test, y_test = _preprocess_svm(X_test, y_test, 4)
            X_test, y_test = shuffle(X_test, y_test)

            samples = X_test[:50]
            # (100, 14, 51)

            # samples = np.reshape(samples, (100, 714)).astype(np.float64)

            df = pd.DataFrame(samples)
            # samples = df.sample(100)

            print("Calculating for", model['path'])
            vals = calculate_h_values(df, model['model'], "results/h-vals/swat/" + "hvals-" + model['path'] + str(i))
