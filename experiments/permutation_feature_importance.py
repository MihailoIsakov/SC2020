"""
Runs recursive feature elimination on the dataset.
We implement our own recursive feature elimination because we don't neccessarily trust
that *.coef_ and *.feature_importance_ are representative of true impact on the model.

We use recursive permutation feature importance, where we first drop correlated features,
then train a classifier, find least important features, drop them, and repeat the process.
"""
import sys
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
# Import root module
sys.path.insert(0, '../')
import dataset

import seaborn as sns
sns.set()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

# This will allow memoization:
memory = joblib.Memory(location='.cache', verbose=0)


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the percentage difference between the prediction and the true value.
    Since y_true represents the log10 value of the raw data, we raise 10 to both
    y_pred and y_true, and divide them. 
    """
    percentage_error = 10**(y_pred - y_true)
    return np.mean(np.abs(percentage_error))


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


@memory.cache
def recursive_permutation_feature_elimination():
    """
    Repeatedly runs permutation feature importance, determines the least important features, drops them,
    and repeats the process until only one feature is left.
    """
    random.seed(0)
    np.random.seed(0)

    df, _ = dataset.default_dataset(paths=["../data/anonimized_io.csv"])

    # Extract IO throughput 
    IO_log10_throughput = df.POSIX_LOG10_agg_perf_by_slowest
    df.drop(columns=["POSIX_RAW_agg_perf_by_slowest", "POSIX_LOG10_agg_perf_by_slowest", "LOG10_runtime"], inplace=True)

    # Drop nonessential features 
    log_columns = list(set([x for x in df.columns if "LOG10" in x or "perc" in x.lower()]))
    df = df[log_columns]

    # Take a subset of the dataset to speed up computation
    # sample = random.sample(range(df.shape[0]), 100000)
    # df = df.iloc[sample]
    # IO_log10_throughput = IO_log10_throughput[sample]

    mape_results = []
    dropped_features = [] 

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, IO_log10_throughput, test_size=0.3, shuffle=True)

    while X_train.shape[1] > 0:     
        print("Dataset size: {}".format(X_train.shape))

        model = xgb.XGBRegressor(obj=huber_approx_obj)
        model.fit(X_train, y_train, eval_metric=huber_approx_obj)

        mape_results.append(mean_absolute_percentage_error(y_test, model.predict(X_test)))
        print("Model achieved MAPE value of {}".format(mape_results[-1]))

        result = permutation_importance(model, X_train, y_train, n_repeats=5, n_jobs=8, random_state=0xdeadbeef)

        least_important_feature_index = np.argmin(result.importances_mean)
        dropped_features.append(X_train.columns[least_important_feature_index])

        print("Dropping feature {}".format(X_train.columns[least_important_feature_index]))
        X_train = X_train.drop(columns=X_train.columns[least_important_feature_index])
        X_test  = X_test.drop (columns=X_test.columns [least_important_feature_index])

    return mape_results, dropped_features


def calculate_feature_importance():
    # Run recursive feature elimination
    mape_results_xgb, dropped_features_xgb = recursive_permutation_feature_elimination()

    # Create a dataframe of the right size
    df = pd.DataFrame(0, index=np.arange(len(mape_results_xgb)), columns=["mape_xgb", "features_xgb"])

    # Fill the dataframe 
    df.mape_xgb     = mape_results_xgb
    df.features_xgb = dropped_features_xgb

    # Plot results 
    MOST_IMPORTANT = 15 # number of most important features to print 
    
    fig, ax1 = plt.subplots(figsize=(3, 3))
    plt.subplots_adjust(left=0.5)
    plt.xlabel("Relative error", fontsize=14)
    plt.title("Feature importance in descending order", fontsize=16, loc='right')

    # Plots 
    ax1.set_xlim(1, 2)
    ax1.plot(df.mape_xgb[-MOST_IMPORTANT:], range(MOST_IMPORTANT), label="Gradient Boosting regressor")

    # X ticks
    ax1.set_xticks([])
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["{}%".format(x*100) for x in [1, 2, 3]], fontsize=12)

    # Remap labels 
    mapping = {
        "POSIX_LOG10_SEEKS"           : "Number of seek() operations",
        "POSIX_SIZE_WRITE_0_100_PERC" : "Ratio of writes in the 0-100B range vs. all accesses",
        "POSIX_ACCESS3_COUNT_PERC"    : "Percentage of accesses in 3rd most common access size",
        "POSIX_shared_files_perc"     : "Ratio of shared files to all files",
        "POSIX_BYTES_WRITTEN_PERC"    : "Ratio of bytes written vs. all bytes",
        "POSIX_SIZE_READ_1K_10K_PERC" : "Ratio of reads in the 1KiB-10KiB range vs. all accesses",
        "POSIX_LOG10_OPENS"           : "Number of open() operations",
        "POSIX_CONSEC_READS_PERC"     : "Percentage of reads that are consecutive", 
        "LOG10_nprocs"                : "Number of processes",
        "POSIX_shared_bytes_perc"     : "Ratio of bytes written/read from shared files vs. all files",
        "POSIX_CONSEC_WRITES_PERC"    : "Percentage of writes that are consecutive", 
        "POSIX_LOG10_STATS"           : "Number of stat() operations",
        "POSIX_LOG10_total_files"     : "Number of files", 
        "POSIX_LOG10_total_accesses"  : "Number of read / write accesses",
        "POSIX_LOG10_total_bytes"     : "Number of bytes read / written"
    }

    labels = df.features_xgb[-MOST_IMPORTANT:]
    labels = [mapping[l].replace('\n', ' ') for l in labels]

    # Y ticks
    ax1.set_yticks(range(MOST_IMPORTANT))
    ax1.set_yticklabels(labels, fontsize=14)

    # Save to pdf
    plt.show()


if __name__ == "__main__": 
    calculate_feature_importance()
