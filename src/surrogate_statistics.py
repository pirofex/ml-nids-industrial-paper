import joblib
from scipy.stats import mannwhitneyu
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats.mstats import normaltest
from scipy.stats import levene

dists_path = "../results/dists_surrogates"

dsts = joblib.load(dists_path)
stats = []

for model, model_dtsts in dsts.items():
    dsts_true = []
    dsts_false = []
    dsts_variance_true = []
    dsts_variance_false = {}
    last_name = "Bayes"
    current_dsts = []
    for surrogate_dst in model_dtsts:
        for surrogate, dst in surrogate_dst.items():
            if "Original" not in str(surrogate):
                name = surrogate[:len(surrogate) - 2]
                if name != last_name:
                    if last_name == str(model):
                        dsts_variance_true = current_dsts
                    else:
                        dsts_variance_false[last_name] = current_dsts
                    current_dsts = []
                current_dsts.append(dst)
                if str(model) == surrogate[:len(surrogate) - 2]:
                    dsts_true.append(dst)
                else:
                    dsts_false.append(dst)
                last_name = name
    if last_name == str(model):
        dsts_variance_true = current_dsts
    else:
        dsts_variance_false[last_name] = current_dsts

    #for elem in dsts_variance_test:
        #print(f"namel test: {normaltest(elem)}")
    #print(f"levene: {levene(*dsts_variance_test)}")
    assert len(dsts_false) == 30
    assert len(dsts_true) == 5
    assert len(dsts_variance_true) == 5
    assert len(dsts_variance_false) == 6
    #stat = mannwhitneyu(dsts_true, dsts_false, alternative="less")
    #stat = f_oneway(*dsts_variance_test)
    curr_stats = {}
    for curr_model, model_dsts in dsts_variance_false.items():
        #stat = mannwhitneyu(dsts_true, model_dsts, alternative="less")
        stat = kruskal(dsts_true, model_dsts)
        curr_stats[curr_model] = stat
    #stat = kruskal(*dsts_variance_test)
    stats.append({model: curr_stats})

print(stats)
