import joblib


def gas_plot_results():
    """Plots the results of the variance analysis (H-Statistic) for the gas pipeline dataset.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    names = [
        'var-rf-10-',
        'var-rf-50-',
        'var-rf-100-',
        'var-rf-250-',
        'var-rf-500-',
        'var-rf-1000-',
        'var-rf-10000-',
    ]

    means = []
    for name in names:
        stds = []

        hvals0 = joblib.load("results/h-vals/" + name + "0")
        hvals1 = joblib.load("results/h-vals/" + name + "1")
        hvals2 = joblib.load("results/h-vals/" + name + "2")

        for a, b, c in zip(hvals0['h_Val'].values, hvals1['h_Val'].values, hvals2['h_Val'].values):
            stds.append(np.std(np.nan_to_num([a, b, c])))

        means.append(np.mean(stds))

    print(means)
    names = ["10", "50", "100", "250", "500", "1000", "10000"]

    import matplotlib.pyplot as plt
    plt.bar(names, means)
    plt.ylabel('mean standard deviation')
    plt.xlabel('amount of samples')
    plt.show()


def swat_plot_results():
    """Plots the results of the variance analysis (H-Statistic) for the SWaT dataset.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib

    names = [
        'var-rf-10-',
        'var-rf-50-',
        'var-rf-100-',
        'var-rf-250-'
    ]

    means = []
    for name in names:
        stds = []

        hvals0 = joblib.load("results/h-vals/swat/" + name + "0")
        hvals1 = joblib.load("results/h-vals/swat/" + name + "1")
        hvals2 = joblib.load("results/h-vals/swat/" + name + "2")

        for a, b, c in zip(hvals0['h_Val'].values, hvals1['h_Val'].values, hvals2['h_Val'].values):
            stds.append(np.std(np.nan_to_num([a, b, c])))

        means.append(np.mean(stds))

    print(means)
    names = ["10", "50", "100", "250"]

    import matplotlib.pyplot as plt
    plt.bar(names, means)
    plt.ylabel('mean standard deviation')
    plt.xlabel('amount of samples')
    plt.show()


def gas_pipeline():
    """Calculates the H-Statistic values of the RF model based
       on the gas pipeline dataset three times for different
       samples sizes and saves them to disk.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from dataset import load_full_data
    from model_creator import load_rf
    from h_statistic import calculate_h_values
    import numpy as np

    import joblib
    import itertools

    samples_sizes = [10, 50, 100, 250, 500, 1000, 10000]

    for sample_size in samples_sizes:
        for i in range(3):
            model = load_rf()
            data = load_full_data()
            samples = data['X_test'].sample(sample_size)

            calculate_h_values(samples, model['box'], "results/h-vals/" + "var-rf-" + str(sample_size) + "-" + str(i))


def swat():
    """Calculates the H-Statistic values of the RF model based
       on the SWaT dataset three times for different
       samples sizes and saves them to disk.
    """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from swat_data import swat_load_data, swat_get_all
    from swat_models import swat_rf_model
    from h_statistic import calculate_h_values
    import numpy as np

    import joblib
    import itertools

    samples_sizes = [10, 50, 100, 250, 500, 1000, 10000]

    for sample_size in samples_sizes:
        for i in range(3):
            model = swat_rf_model()
            data = swat_load_data()
            X_train, y_train, X_test, y_test = swat_get_all(data)
            samples = X_train.sample(sample_size)
            del samples['Timestamp']
            calculate_h_values(samples, model, "results/h-vals/swat/" + "var-rf-" + str(sample_size) + "-" + str(i))


if __name__ == '__main__':
    swat_plot_results()
