import joblib
import os


def gas_eval():
    """Loads the calculated H-Statistic values of the gas pipeline models and plots
       them as scatterplot.
    """

    names = [
        'hvals-rules',
        'hvals-logistic-regression',
        'hvals-bayes',
        'hvals-nn',
        'hvals-nn-linear',
        'hvals-svm',
        'hvals-svm-linear',
        'hvals-rf'
    ]

    import numpy as np
    import matplotlib.pyplot as plt

    labels = ["rules", "logistic", "bayes", "NN", "NN Linear",
              "SVM", "SVM Linear",
              "RF"]

    colors = ["red", "blue", "gold", "black", "blue", "purple", "orange", "green", "red"]

    vals = []
    for name in names:
        hvals0 = joblib.load("results/h-vals/" + name + "0")
        vals.append(hvals0['h_Val'].values)

    width = 0.4
    fig, ax = plt.subplots()
    for i, l in enumerate(labels):
        data = np.random.rayleigh(scale=1, size=(vals[i].shape[0], 2))
        vals[i][np.isnan(vals[i])] = 0
        x = np.ones(data.shape[0]) * i + (np.random.rand(data.shape[0]) * width - width / 2.)
        d = vals[i]
        d[np.isnan(d)] = 0
        ax.scatter(x, d, color=colors[i], s=25)
        mean = np.mean(d)
        print(f"{l}: {mean}")
        ax.plot([i - width / 2., i + width / 2.], [mean, mean], color="k")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    plt.show()

def swat_a45_eval():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st

    labels = ["Bayes", "LR", "NN", "NN linear", "RF", "SVM", "SVM linear"]
    colors = ["red", "red", "red", "red", "red", "red", "red"]

    vals = []
    for name in [
        "../results/h-vals/swat-a45/a45_hvals-bayes0",
        "../results/h-vals/swat-a45/a45_hvals-lr0",
        "../results/h-vals/swat-a45/a45_hvals-nn0",
        "../results/h-vals/swat-a45/a45_hvals-nn_linear0",
        "../results/h-vals/swat-a45/a45_hvals-rf0",
        "../results/h-vals/swat-a45/a45_hvals-svm0",
        "../results/h-vals/swat-a45/a45_hvals-svm_linear0",
    ]:
        hvals0 = joblib.load(name)
        vals.append(hvals0['h_Val'].values)

    width = 0.4
    fig, ax = plt.subplots()
    for i, l in enumerate(labels):
        data = np.random.rayleigh(scale=1, size=(vals[i].shape[0], 2))
        vals[i][np.isnan(vals[i])] = 0
        x = np.ones(data.shape[0]) * i + (np.random.rand(data.shape[0]) * width - width / 2.)
        d = vals[i]
        d[np.isnan(d)] = 0
        ax.scatter(x, d, color=colors[i], s=25)
        mean = np.mean(d)
        interval = st.t.interval(0.95, len(d)-1, loc=np.mean(d), scale=st.sem(d))
        print(f"{l}: mean = {str(mean)}, interval = {str(interval)}")
        ax.plot([i - width / 2., i + width / 2.], [mean, mean], color="k")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    plt.show()

if __name__ == '__main__':
    """Loads the calculated H-Statistic values of the SWaT models and plots
       them as scatterplot.
    """
    swat_a45_eval()
    raise Exception


    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st

    #labels = ["SVM", "SVM linear", "NN", "NN linear", "rf", "bayes", "lr"]
    #colors = ["red", "red", "red", "red", "red", "red", "red"]
    labels = ["Bayes", "LR", "RF", "SVM", "SVM-Linear", "NN", "NN-Linear"]
    colors = ["red", "red", "red", "red", "red", "red", "red"]

    vals = []
    for name in [
        #"../results/h-vals/gas-pipeline/a45_hvals-bayes1",
        #"../results/h-vals/gas-pipeline/hvals-logistic-regression1",
        #"../results/h-vals/gas-pipeline/hvals-rf1",
        #"../results/h-vals/gas-pipeline/hvals-svm1",
        #"../results/h-vals/gas-pipeline/hvals-svm-linear1",
        #"../results/h-vals/gas-pipeline/hvals-nn1",
        #"../results/h-vals/gas-pipeline/hvals-nn-linear1",
        #"../results/h-vals/swat/a45_hvals-bayes1",
        #"../results/h-vals/swat/hvals-logistic-regression1",
        #"../results/h-vals/swat/hvals-rf_new1",
        "../results/h-vals/swat/hvals-svm",
        "../results/h-vals/swat/hvals-svm-linear",
        #"../results/h-vals/swat/hvals-nn1",
        #"../results/h-vals/swat/hvals-nn-linear1",
        ]:
        hvals0 = joblib.load(name)
        vals.append(hvals0['h_Val'].values)

    width = 0.4
    fig, ax = plt.subplots()
    for i, l in enumerate(labels):
        data = np.random.rayleigh(scale=1, size=(vals[i].shape[0], 2))
        vals[i][np.isnan(vals[i])] = 0
        x = np.ones(data.shape[0]) * i + (np.random.rand(data.shape[0]) * width - width / 2.)
        d = vals[i]
        d[np.isnan(d)] = 0
        ax.scatter(x, d, color=colors[i], s=25)
        mean = np.mean(d)
        interval = st.t.interval(0.95, len(d)-1, loc=np.mean(d), scale=st.sem(d))
        print(f"{l}: mean = {str(mean)}, interval = {str(interval)}")
        ax.plot([i - width / 2., i + width / 2.], [mean, mean], color="k")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    plt.show()
