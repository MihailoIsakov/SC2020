"""
This script aims to answer:
    1. What is the best possible relative MAE error of global models? 
    2. Do local models perform better than the global models?
"""
import sys
import joblib
import numpy as np
import sklearn
import sklearn.cluster
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
# Import root module
sys.path.insert(0, '../')
import dataset

# Seaborn pretty plots
seaborn.set()
# This will allow memoization:
memory = joblib.Memory(location='.cache', verbose=0)


def huber_approx_obj(y_pred, y_test):
    """
    Huber loss, adapted from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    """
    d = y_pred - y_test 
    h = 5  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def prediction_error(X, y, obj_function=huber_approx_obj):
    """
    Train GBMs to predict y from X.
    Use obj_function during training, and test_error_function for the test evaluation.
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

    regressor = xgb.XGBRegressor(obj=huber_approx_obj)
    regressor.fit(X_train, y_train, eval_metric=huber_approx_obj)
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)

    return np.abs(y_train - y_pred_train), np.abs(y_test - y_pred_test)


@memory.cache
def get_cluster_results(df, input_columns, epsilons, MIN_CLUSTER_SIZE):
    cluster_errors, cluster_test_points, cluster_sizes, cluster_eps = [], [], [], []

    for eps in epsilons:
        print("Running DBSCAN with eps={}".format(eps))
        clusterer = sklearn.cluster.DBSCAN(eps=eps, metric='manhattan', n_jobs=8)
        clusterer.fit(df[input_columns])

        print("Found {} clusters".format(len(set(clusterer.labels_))-1))
        for cluster_idx in set(clusterer.labels_).difference([-1]):
            if np.sum(clusterer.labels_ == cluster_idx) > MIN_CLUSTER_SIZE: 
                cluster_size = np.sum(clusterer.labels_ == cluster_idx)
                cluster_train_errors, cluster_test_errors = prediction_error(df.iloc[clusterer.labels_ == cluster_idx][input_columns], df.iloc[clusterer.labels_ == cluster_idx].POSIX_LOG10_agg_perf_by_slowest)

                cluster_errors      += cluster_train_errors.tolist() + cluster_test_errors.tolist()
                cluster_test_points += [False] * len(cluster_train_errors) + [True] * len(cluster_test_errors)
                cluster_sizes       += [cluster_size] * cluster_size
                cluster_eps         += [eps] * cluster_size

    results = pd.DataFrame({"jobs_cluster_size": cluster_sizes, 
                            "job_errors":        cluster_errors, 
                            "job_in_test_set":   cluster_test_points,
                            "job_eps":           cluster_eps})

    return results


def main(MIN_CLUSTER_SIZE=100):
    df, clusterer = dataset.default_dataset(paths=["../data/anonimized_io.csv"])
    input_columns = set([c for c in df.columns if 'perc' in c.lower() or 'LOG10' in c]).difference(["POSIX_LOG10_agg_perf_by_slowest"])

    # Hand selected these to show good gradients, since the dataset is finicky.
    # small changes in epsilon lead to large changes in the number of clusters.
    epsilons      = [9.5,  7,  5,  2.1 ]
    cluster_sizes = [10,  79, 267, 1077]

    results = get_cluster_results(df, input_columns, epsilons, MIN_CLUSTER_SIZE)
    global_train_errors, global_test_errors = prediction_error(df[input_columns], df.POSIX_LOG10_agg_perf_by_slowest)
    global_results = pd.DataFrame({"jobs_cluster_size": [df.shape[0]] * (len(global_train_errors) + len(global_test_errors)), 
                                   "job_errors":        list(global_train_errors) + list(global_test_errors), 
                                   "job_in_test_set":   [False] * len(global_train_errors) + [True] * len(global_test_errors),
                                   "job_eps":           [1000] * (len(global_train_errors) + len(global_test_errors))})
    results = results.append(global_results)

    # Let's rescale the errors so that they represent real ratios and not logarithmic differences
    results.job_errors = 10**results.job_errors


    #
    # Plotting 
    #
    fig = plt.figure(figsize=(15, 8))
    spec = fig.add_gridspec(ncols=2, nrows=2)

    ax2 = fig.add_subplot(spec[0, 0])
    ax4 = fig.add_subplot(spec[0, 1], sharey=ax2)

    seaborn.boxplot(data=results[results.job_in_test_set == False], x="job_eps", y="job_errors", ax=ax2)
    ax2.set_title("Cluster Training Errors")
    ax2.set_xlabel("Number of clusters")
    ax2.set_xticklabels(list(reversed(cluster_sizes)) + ["Global"])
    ax2.set_yscale('log')

    seaborn.boxplot(data=results[results.job_in_test_set], x="job_eps", y="job_errors", ax=ax4)
    ax4.set_title("Cluster Test Errors")
    ax4.set_xlabel("Number of clusters")
    ax4.set_xticklabels(list(reversed(cluster_sizes)) + ["Global"])

    ax2.set_ylabel("Average Prediction Ratio Error")
    ax4.set_ylabel("Average Prediction Ratio Error")
    ax2.set_yticks(np.arange(1, 3, 0.1), minor=True)
    ax4.set_yticks(np.arange(1, 3, 0.1), minor=True)
    ax2.grid(axis='x', which='both')
    ax4.grid(axis='x', which='both')

    # Plot the cumulative histograms results[results.job_in_test_set].job_errors
    for eps in reversed([1000] + epsilons):
        ax_cm = fig.add_subplot(spec[1, 0])
        plt.xlim(1, 2)
        plt.ylim(0, 1.2)

        x = results[np.logical_and(results.job_in_test_set == False, results.job_eps==eps)].job_errors
        x = x[x < 3]
        seaborn.distplot(x, norm_hist=True, hist_kws={'cumulative': True, 'histtype': 'step', 'alpha': 1}, ax=ax_cm)

        plt.legend(list(reversed(cluster_sizes)) + ["Global"])
        ax_cm.set_xlabel("Average Prediction Ratio Error")
        ax_cm.set_ylabel("Percentage of jobs")

        ax_cm = fig.add_subplot(spec[1, 1])
        plt.xlim(1, 2)
        plt.ylim(0, 1.2)

        x = results[np.logical_and(results.job_in_test_set, results.job_eps==eps)].job_errors
        x = x[x < 3]
        seaborn.distplot(x, norm_hist=True, hist_kws={'cumulative': True, 'histtype': 'step', 'alpha': 1}, ax=ax_cm)

        plt.legend(list(reversed(cluster_sizes)) + ["Global"])
        ax_cm.set_xlabel("Average Prediction Ratio Error")
        ax_cm.set_ylabel("Percentage of jobs")

    plt.show()


if __name__ == "__main__": 
    main()
