if __name__ == '__main__':
    import joblib
    from swat_models import *
    from swat_data import *
    from dataset import *
    from scipy.io import arff
    from model_creator import *
    from h_statistic import *
    from pdp import *
    import matplotlib.pyplot as plt

    # Create gas pipeline dataset
    data_creator = DatasetCreator()
    data = data_creator.full_data("data/IanArffDataset.arff")

    # Shortcut after dataset has been created
    # data = load_full_data()

    # Train gas pipeline models
    model_creator = ModelCreator()
    rf = model_creator.train_rf(data)
    # nn = model_creator.train_nn(data)
    svm = model_creator.train_svm(data)

    # Shortcut after models have been trained
    # rf = load_rf()
    # svm = load_svm_linear()

    # Create one- and two-dimensional PDP plots for specific model features
    plot_pdp(data['X_test'].sample(500), rf['box'], 'cycle time')
    plot_interaction_pdp(data['X_test'].sample(500), rf['box'], ['cycle time', 'function'])
    plt.show()

    # Calculate H-Statistics for the models
    for model in [svm]:
        samples = data['X_test_norm'].sample(500)
        print("Calculating for", model['type'])
        calculate_h_values(samples, model['box'], "hvals-" + model['type'])

    for model in [rf]:
        samples = data['X_test'].sample(500)
        print("Calculating for", model['type'])
        calculate_h_values(samples, model['box'], "hvals-" + model['type'])

    # Plot H-Statistic results
    import numpy as np
    import matplotlib.pyplot as plt

    labels = ["RF", "SVM (linear)"]
    colors = ["red", "blue"]

    vals = []
    for name in ["hvals-rf", "hvals-svm-linear"]:
        hvals0 = joblib.load(name)
        vals.append(hvals0['h_Val'].values)
 
    width = 0.4
    fig, ax = plt.subplots()
    for i, l in enumerate(labels):
        data = np.random.rayleigh(scale=1, size=(vals[i].shape[0],2))
        vals[i][np.isnan(vals[i])] = 0
        x = np.ones(data.shape[0])*i + (np.random.rand(data.shape[0])*width-width/2.)
        d = vals[i]
        d[np.isnan(d)] = 0
        ax.scatter(x, d, color=colors[i], s=25)
        mean = np.mean(d)
        ax.plot([i-width/2., i+width/2.],[mean,mean], color="k")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    plt.show()
