import joblib

if __name__ == '__main__':
    """Calculates the similarity of H-Statistics between the original model
       and the surrogate models and prints the distance to the console.
       In addition, the H-Statistics for each model are plotted as scatterplot.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # labels = ["RF (original)", "Surrogate RF (Testset)", "Surrogate RF (KDE)"]
    # labels = ["Logistic (original)", "Surrogate Logistic (original)"]
    labels = [
        "Original",
        "Bayes 1",
        "Bayes 2",
        "Bayes 3",
        "Bayes 4",
        "Bayes 5",
        "LR 1",
        "LR 2",
        "LR 3",
        "LR 4",
        "LR 5",
        "NN 1",
        "NN 2",
        "NN 3",
        "NN 4",
        "NN 5",
        "NN lin. 1",
        "NN lin. 2",
        "NN lin. 3",
        "NN lin. 4",
        "NN lin. 5",
        "RF 1",
        "RF 2",
        "RF 3",
        "RF 4",
        "RF 5",
        "SVM 1",
        "SVM 2",
        "SVM 3",
        "SVM 4",
        "SVM 5",
        "SVM lin. 1",
        "SVM lin. 2",
        "SVM lin. 3",
        "SVM lin. 4",
        "SVM lin. 5"
    ]
    colors = ["red" for i in range(len(labels))]
    dists = {}
    for original in [
        ["../results/h-vals/gas-pipeline/hvals-bayes-norm1", "Bayes"],
        ["../results/h-vals/gas-pipeline/hvals-logistic-regression1", "LR"],
        ["../results/h-vals/gas-pipeline/hvals-nn1", "NN"],
        ["../results/h-vals/gas-pipeline/hvals-nn-linear1", "NN lin."],
        ["../results/h-vals/gas-pipeline/hvals-rf-norm1", "RF"],
        ["../results/h-vals/gas-pipeline/hvals-svm1", "SVM"],
        ["../results/h-vals/gas-pipeline/hvals-svm-linear1", "SVM lin."]
    ]:
        dists[original[1]] = []
        vals = []
        print("\r\n## " + str(original[0]) + "\r\n")
        for name in [
            original[0],
            "../results/h-vals/gas-surrogates_nol/hvals-bayes-01",
            "../results/h-vals/gas-surrogates_nol/hvals-bayes-11",
            "../results/h-vals/gas-surrogates_nol/hvals-bayes-21",
            "../results/h-vals/gas-surrogates_nol/hvals-bayes-31",
            "../results/h-vals/gas-surrogates_nol/hvals-bayes-41",
            "../results/h-vals/gas-surrogates_nol/hvals-lr-01",
            "../results/h-vals/gas-surrogates_nol/hvals-lr-11",
            "../results/h-vals/gas-surrogates_nol/hvals-lr-21",
            "../results/h-vals/gas-surrogates_nol/hvals-lr-31",
            "../results/h-vals/gas-surrogates_nol/hvals-lr-41",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-01",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-11",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-21",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-31",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-41",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-linear-01",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-linear-11",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-linear-21",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-linear-31",
            "../results/h-vals/gas-surrogates_nol/hvals-nn-linear-41",
            "../results/h-vals/gas-surrogates_nol/hvals-rf-01",
            "../results/h-vals/gas-surrogates_nol/hvals-rf-11",
            "../results/h-vals/gas-surrogates_nol/hvals-rf-21",
            "../results/h-vals/gas-surrogates_nol/hvals-rf-31",
            "../results/h-vals/gas-surrogates_nol/hvals-rf-41",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-01",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-11",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-21",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-31",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-41",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-linear-01",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-linear-11",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-linear-21",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-linear-31",
            "../results/h-vals/gas-surrogates_nol/hvals-svm-linear-41",
        ]:
            hvals0 = joblib.load(name)
            vals.append(hvals0['h_Val'].values)

        from scipy.spatial import distance

        original_rf_vals = np.nan_to_num(vals[0])

        for label, val in zip(labels, vals):
            normalized_vals = np.nan_to_num(val)
            dst = distance.euclidean(original_rf_vals, normalized_vals)

            print(label, dst)
            dists[original[1]].append({label: dst})

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
            ax.plot([i - width / 2., i + width / 2.], [mean, mean], color="k")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

        # plt.show()
    joblib.dump(dists, "../results/dists_surrogates")