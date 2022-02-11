if __name__ == '__main__':
    """In this file the H-Statistic values for the surrogate models
       are calculated. Almost the same as in the file 'h_statistic.py'.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from dataset import *
    from model_creator import *
    from gp_rule_model import *
    from h_statistic import *

    import joblib
    import itertools

    for k in range(5):
        models = [
            #{'model': joblib.load(f"../results/surrogates_nol/lr-{k}"), 'path': f'lr-{k}'},
            #{'model': joblib.load(f"../results/surrogates_nol/bayes-{k}"), 'path': f'nn-{k}'},
            {'model': joblib.load(f"../results/surrogates_nol/rf-{k}"), 'path': f'rf-{k}'},
            {'model': joblib.load(f"../results/surrogates_nol/svm-{k}"), 'path': f'svm-{k}'},
            {'model': joblib.load(f"../results/surrogates_nol/svm-linear-{k}"), 'path': f'svm-linear-{k}'},
            {'model': joblib.load(f"../results/surrogates_nol/nn-{k}"), 'path': f'nn-{k}'},
            {'model': joblib.load(f"../results/surrogates_nol/nn-linear-{k}"), 'path': f'nn-linear-{k}'},
        ]

        data = load_full_data()
        X_test = data['X_test_norm']

        scaler = preprocessing.StandardScaler()
        X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
        samples = X_test.sample(500)

        for model in models:

            for i in range(1):
                i = i + 1

                print("Calculating for", model['path'])

                if model['path'] == 'logistic':
                    calculate_h_values(samples, model['model'],
                                       "../results/h-vals/gas-surrogates_nol/" + "hvals-" + model['path'] + str(i))
                else:
                    calculate_h_values(samples, model['model']['box'],
                                       "../results/h-vals/gas-surrogates_nol/" + "hvals-" + model['path'] + str(i))
