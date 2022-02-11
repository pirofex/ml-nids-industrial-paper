from pdpbox.pdp import pdp_isolate, PDPInteract
from pdpbox.utils import (_check_model, _check_dataset, _check_percentile_range, _check_feature,
                          _check_grid_type, _check_memory_limit, _make_list,
                          _calc_memory_usage, _get_grids, _get_grid_combos, _check_classes)
from joblib import Parallel, delayed
from sklearn import preprocessing
import numpy as np
import pandas as pd
import itertools
import math
import joblib


# Code from here: https://blog.macuyiko.com/post/2019/discovering-interaction-effects-in-ensemble-models.html

def _calc_ice_lines_inter(feature_grids_combo, data, model, model_features, n_classes, feature_list,
                          predict_kwds, data_transformer, unit_test=False):
    """Apply predict function on a grid combo

    Returns
    -------
    Predicted result on this feature_grid
    """

    _data = data.copy()

    for idx in range(len(feature_list)):
        _data[feature_list[idx]] = feature_grids_combo[idx]

    if n_classes == 0:
        predict = model.predict
    else:
        predict = model.predict_proba

    preds = predict(_data[model_features], **predict_kwds)
    grid_result = _data[feature_list].copy()

    if n_classes == 0:
        grid_result['preds'] = preds
    elif n_classes == 2:
        grid_result['preds'] = preds[:, 1]
    else:
        for n_class in range(n_classes):
            grid_result['class_%d_preds' % n_class] = preds[:, n_class]

    return grid_result


def pdp_multi_interact(model, dataset, model_features, features,
                       num_grid_points=None, grid_types=None, percentile_ranges=None, grid_ranges=None,
                       cust_grid_points=None,
                       cust_grid_combos=None, use_custom_grid_combos=False,
                       memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None):
    def _expand_default(x, default, length):
        if x is None:
            return [default] * length
        return x

    def _get_grid_combos(feature_grids, feature_types):
        grids = [list(feature_grid) for feature_grid in feature_grids]
        for i in range(len(feature_types)):
            if feature_types[i] == 'onehot':
                grids[i] = np.eye(len(grids[i])).astype(int).tolist()
        return np.stack(np.meshgrid(*grids), -1).reshape(-1, len(grids))

    if predict_kwds is None:
        predict_kwds = dict()

    nr_feats = len(features)

    # check function inputs
    n_classes, predict = _check_model(model=model)
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    # prepare the grid
    pdp_isolate_outs = []
    if use_custom_grid_combos:
        grid_combos = cust_grid_combos
        feature_grids = []
        feature_types = []
    else:
        num_grid_points = _expand_default(x=num_grid_points, default=10, length=nr_feats)
        grid_types = _expand_default(x=grid_types, default='percentile', length=nr_feats)
        for i in range(nr_feats):
            _check_grid_type(grid_type=grid_types[i])

        percentile_ranges = _expand_default(x=percentile_ranges, default=None, length=nr_feats)
        for i in range(nr_feats):
            _check_percentile_range(percentile_range=percentile_ranges[i])

        grid_ranges = _expand_default(x=grid_ranges, default=None, length=nr_feats)
        cust_grid_points = _expand_default(x=cust_grid_points, default=None, length=nr_feats)

        _check_memory_limit(memory_limit=memory_limit)

        pdp_isolate_outs = []
        for idx in range(nr_feats):
            pdp_isolate_out = pdp_isolate(
                model=model, dataset=_dataset, model_features=model_features, feature=features[idx],
                num_grid_points=num_grid_points[idx], grid_type=grid_types[idx],
                percentile_range=percentile_ranges[idx],
                grid_range=grid_ranges[idx], cust_grid_points=cust_grid_points[idx], memory_limit=memory_limit,
                n_jobs=n_jobs, predict_kwds=predict_kwds, data_transformer=data_transformer)
            pdp_isolate_outs.append(pdp_isolate_out)

        if n_classes > 2:
            feature_grids = [pdp_isolate_outs[i][0].feature_grids for i in range(nr_feats)]
            feature_types = [pdp_isolate_outs[i][0].feature_type for i in range(nr_feats)]
        else:
            feature_grids = [pdp_isolate_outs[i].feature_grids for i in range(nr_feats)]
            feature_types = [pdp_isolate_outs[i].feature_type for i in range(nr_feats)]

        grid_combos = _get_grid_combos(feature_grids, feature_types)

    feature_list = []
    for i in range(nr_feats):
        feature_list.extend(_make_list(features[i]))

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(grid_combos), n_jobs=n_jobs, memory_limit=memory_limit)

    grid_results = Parallel(n_jobs=true_n_jobs)(delayed(_calc_ice_lines_inter)(
        grid_combo, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
        feature_list=feature_list, predict_kwds=predict_kwds, data_transformer=data_transformer)
                                                for grid_combo in grid_combos)

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    pdp_interact_params = {'n_classes': n_classes,
                           'features': features,
                           'feature_types': feature_types,
                           'feature_grids': feature_grids}
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ['class_%d_preds' % n_class]].rename(
                columns={'class_%d_preds' % n_class: 'preds'})
            pdp_interact_out.append(
                PDPInteract(which_class=n_class,
                            pdp_isolate_outs=[pdp_isolate_outs[i][n_class] for i in range(nr_feats)],
                            pdp=_pdp, **pdp_interact_params))
    else:
        pdp_interact_out = PDPInteract(
            which_class=None, pdp_isolate_outs=pdp_isolate_outs, pdp=pdp, **pdp_interact_params)

    return pdp_interact_out


def center(arr): return arr - np.mean(arr)


def _calc_comb(selectedfeatures, n, f_vals, num_grid_points, data_grid, use_data_grid, mdl, X, features, grid):
    print("Starting combinations for feature tuple", n)
    for subsetfeatures in itertools.combinations(selectedfeatures, n):
        # print("subsetfeat", str(subsetfeatures))
        if use_data_grid:
            data_grid = X[list(subsetfeatures)].values
        p_partial = pdp_multi_interact(mdl, X, features, subsetfeatures,
                                       num_grid_points=[num_grid_points] * len(selectedfeatures),
                                       cust_grid_combos=data_grid,
                                       use_custom_grid_combos=use_data_grid)
        p_joined = pd.merge(grid, p_partial.pdp, how='left')
        f_vals[tuple(subsetfeatures)] = center(p_joined.preds.values)
    print("Finished combinations for feature tuple", n)


def compute_f_vals(mdl, X, features, selectedfeatures, num_grid_points=10, use_data_grid=False, data_grid=0,
                   transformer=None):
    from multiprocessing import Manager
    manager = Manager()
    f_vals = manager.dict()

    # data_grid = None
    # if use_data_grid:
    #    data_grid = X[selectedfeatures].values
    print("PDP for full feature set ...")
    # Calculate partial dependencies for full feature set
    p_full = pdp_multi_interact(mdl, X, features, selectedfeatures,
                                num_grid_points=[num_grid_points] * len(selectedfeatures),
                                cust_grid_combos=data_grid,
                                data_transformer=transformer,
                                use_custom_grid_combos=use_data_grid)
    f_vals[tuple(selectedfeatures)] = center(p_full.pdp.preds.values)
    grid = p_full.pdp.drop('preds', axis=1)
    # Calculate partial dependencies for [1..SFL-1]

    import multiprocessing
    processes = []

    for n in range(1, len(selectedfeatures)):
        print(f"caluclation for {selectedfeatures[n]}")
        _calc_comb(selectedfeatures, n, f_vals, num_grid_points, data_grid, use_data_grid, mdl, X, features, grid)
    return f_vals

    # for n in range(1, len(selectedfeatures)):
    #    p = multiprocessing.Process(target=_calc_comb, args=(
    #    selectedfeatures, n, f_vals, num_grid_points, data_grid, use_data_grid, mdl, X, features, grid))
    #    processes.append(p)
    #    p.start()

    # for process in processes:
    #    process.join()

    # return f_vals


def compute_h_val(f_vals, selectedfeatures):
    denom_els = f_vals[tuple(selectedfeatures)].copy()
    numer_els = f_vals[tuple(selectedfeatures)].copy()
    sign = -1.0
    for n in range(len(selectedfeatures) - 1, 0, -1):
        for subfeatures in itertools.combinations(selectedfeatures, n):
            numer_els += sign * f_vals[tuple(subfeatures)]
        sign *= -1.0
    numer = np.sum(numer_els ** 2)
    denom = np.sum(denom_els ** 2)
    return math.sqrt(numer / denom) if numer < denom else np.nan


def compute_h_val_any(f_vals, allfeatures, selectedfeature):
    otherfeatures = list(allfeatures)
    otherfeatures.remove(selectedfeature)
    denom_els = f_vals[tuple(allfeatures)].copy()
    numer_els = denom_els.copy()
    numer_els -= f_vals[(selectedfeature,)]
    numer_els -= f_vals[tuple(otherfeatures)]
    numer = np.sum(numer_els ** 2)
    denom = np.sum(denom_els ** 2)
    return math.sqrt(numer / denom) if numer < denom else np.nan


###################################
# HERE STARTS MY OWN IMPLEMTATION #
###################################

def calculate_h_values_lstm(df, model, path):
    """Calculates the H-Statistic values for my LSTM.
    Because of more complex dimensions (window size),
    it is necessary to flatten it in order to be able
    to compute the different feature pairs.

    Args:
        df (DataFrame): The samples used for the calculation of the H-Statistics
        model ([type]): The model which predict function is used.
        path (str): The path where the H-Statistic values shall be saved to.
    """

    # (100, 14, 51)

    def my_transformer(d):
        # lstm = joblib.load("results/models/swat/lstm")
        import numpy as np
        n = d.to_numpy()
        r = np.reshape(n, (n.shape[0], 14, 51))
        # asdf = lstm.predict(r)
        return r

    combinations = list(itertools.combinations(df.columns, 2))
    print("Combinations:", len(combinations))
    h_values = pd.DataFrame(columns=['relation', 'h_val'])

    for combi in combinations:
        f_vals = compute_f_vals(model, df, df.columns, [combi[0], combi[1]], use_data_grid=False,
                                transformer=my_transformer)
        h_val = compute_h_val(f_vals, [combi[0], combi[1]])
        print(combi, h_val)

        row = pd.Series([(df.columns.get_loc(combi[0]), df.columns.get_loc(combi[1])), h_val],
                        index=['relation', 'h_Val'])
        h_values = h_values.append(row, ignore_index=True)

    joblib.dump(h_values, path)


def calculate_h_values(df, model, path):
    """Calculates the H-Statistic values for the given model.

    Args:
        df (DataFrame): The samples used for the calculation of the H-Statistics
        model ([type]): The model which predict function is used.
        path (str): The path where the H-Statistic values shall be saved to.
    """
    combinations = list(itertools.combinations(df.columns, 2))
    print("Combinations:", len(combinations))
    h_values = pd.DataFrame(columns=['relation', 'h_val'])

    for combi in combinations:
        f_vals = compute_f_vals(model, df, df.columns, [combi[0], combi[1]], use_data_grid=False)
        h_val = compute_h_val(f_vals, [combi[0], combi[1]])
        print(combi, h_val)

        row = pd.Series([(df.columns.get_loc(combi[0]), df.columns.get_loc(combi[1])), h_val],
                        index=['relation', 'h_Val'])
        h_values = h_values.append(row, ignore_index=True)

    joblib.dump(h_values, path)


def calculate_h_values_swat_svm(df, model, combinations):
    """Calculates the H-Statistic values for the given model and the given combinations.

    Args:
        df (DataFrame): The samples used for the calculation of the H-Statistics
        model ([type]): The model which predict function is used.
        combinations ([type]): The combinations which shall be calculated.

    Returns:
        [type]: [description]
    """
    print("Combinations:", len(combinations))
    h_values = pd.DataFrame(columns=['relation', 'h_val'])

    f_vals = compute_f_vals(model, df, df.columns, [combinations[0], combinations[1]])
    h_val = compute_h_val(f_vals, [combinations[0], combinations[1]])
    print(combinations, h_val)

    row = pd.Series([(combinations[0], combinations[1]), h_val], index=['relation', 'h_Val'])
    h_values = h_values.append(row, ignore_index=True)

    return h_values


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from dataset import *
    from model_creator import *
    from gp_rule_model import *

    import joblib
    import itertools

    # Gas pipeline
    #models = [
        #    {'model': load_rule_model(), 'path': 'rules'},
        #    {'model': load_bayes(), 'path': 'bayes'},
        #    {'model': load_rf(), 'path': 'rf'},
    #    {'model': joblib.load(f"../results/surrogates_nol/bayes-{k}"), 'path': f'bayes-{k}'},
    #    {'model': joblib.load(f"../results/surrogates_nol/rf-{k}"), 'path': f'rf-{k}'},

    #]

    # We need to differentiate between models which take normalized data as input, and
    # models which take not-normalied data as input.
    #models_norm = [
        # {'model': load_logistic_regression(), 'path': 'logistic-regression'},
        # {'model': load_nn(), 'path': 'nn'},
        # {'model': load_nn_linear(), 'path': 'nn-linear'},
        #    {'model': load_svm(), 'path': 'svm'},
        #    {'model': load_svm_linear(), 'path': 'svm-linear'},
        #{'model': joblib.load(f"../results/surrogates_nol/lr-{k}"), 'path': f'lr-{k}'},
        #{'model': joblib.load(f"../results/surrogates_nol/svm-{k}"), 'path': f'svm-{k}'},
        #{'model': joblib.load(f"../results/surrogates_nol/svm-linear-{k}"), 'path': f'svm-linear-{k}'},
        #{'model': joblib.load(f"../results/surrogates_nol/nn-{k}"), 'path': f'nn-{k}'},
        #{'model': joblib.load(f"../results/surrogates_nol/nn-linear-{k}"), 'path': f'nn-linear-{k}'},
        #{'model': joblib.load(f"../results/models/bayes-norm"), 'path': f'bayes-norm'},
        #{'model': joblib.load(f"../results/models/rf-norm"), 'path': f'rf-norm'},
    #]
    models_norm=[]
    models = [
        {'model': joblib.load("../results/models/swat-a45/a45_bayes"), 'path':'bayes'},
        {'model': joblib.load("../results/models/swat-a45/a45_lr"), 'path':'lr'}
    ]

    # SWAT
    import swat_models
    #import swat_data
   # models = [
        #{'model': swat_models.swat_bayes_model(), 'path': 'bayes'},
        #{'model': swat_models.swat_rf_model(), 'path': 'rf_new'},
    #]

    # We need to differentiate between models which take normalized data as input, and
    # models which take not-normalized data as input.
    #models_norm = [
        # {'model': swat_models.swat_lr_model(), 'path': 'logistic-regression'},
        # {'model': swat_models.swat_nn_model(), 'path': 'nn'},
        # {'model': swat_models.swat_nn_linear_model(), 'path': 'nn-linear'},
        # {'model': swat_models.swat_svm_model(), 'path': 'svm'},
        # {'model': swat_models.swat_svm_linear_model(), 'path': 'svm-linear'},
    #]

    for model in models_norm:
        for i in range(1):
            i = i + 1
            # gas pipline
            data = load_full_data()
            samples = data['X_train_norm'].sample(500)

            # swat
            #data = swat_data.swat_load_data()
            #X_train, y_train, X_test, y_test = swat_data.swat_get_all(data)
            #samples = X_train.sample(500)
            #del samples['Timestamp']
            #scaler = preprocessing.StandardScaler()
            #samples[samples.columns] = scaler.fit_transform(samples[samples.columns])

            print("Calculating for", model['path'])
            #calculate_h_values(samples, model['model']['box'],
            #                   "../results/h-vals/swat/" + "hvals-" + model['path'] + str(i))
            calculate_h_values(samples, model['model']['box'],
                               "../results/h-vals/gas-pipeline/" + "hvals-" + model['path'] + str(i))

    for model in models:
        for i in range(1):
            i = i + 1
            # gas pipeline
            #data = load_full_data()
            #samples = data['X_train'].sample(500)

            # swat45
            data = joblib.load("../data/swat/a45")
            samples = data['X_train'].sample(500)

            # swat
            #data = swat_data.swat_load_data()
            #X_train, y_train, X_test, y_test = swat_data.swat_get_all(data)
            #samples = X_train.sample(500)
            #del samples['Timestamp']

            print("Calculating for", model['path'])
            #calculate_h_values(samples, model['model']['box'],
            #                   "../results/h-vals/swat-a45/" + "hvals-" + model['path'] + str(i))
            calculate_h_values(samples, model['model'],
                               "../results/h-vals/swat-a45/" + "hvals-" + model['path'] + str(i))
