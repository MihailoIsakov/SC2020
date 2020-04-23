"""
The Gauge dashboard: a tool for displaying information about a number of clusters.
Should allow exploration of different clustering methods, different types of plots, etc.
all within the same visualization.
"""
import copy
import argparse
import sys
import random
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import logging
import joblib
from pandas.plotting import parallel_coordinates
import shap
# Import root module
sys.path.insert(0, '../')
import dataset
import feature_name_mapping

# This will allow memoization:
memory = joblib.Memory(location='.cache', verbose=0)

# Apply seaborn styles
sns.set()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

# Matplotlib won't plot otherwise for some reason
matplotlib.use('tkagg')


#
# Tick formatters
#
@ticker.FuncFormatter
def perc_formatter(x, pos):
    return "%.0f" % (100*x) + " %"


@ticker.FuncFormatter
def log_formatter(x, pos):
    return r"$10^{%.1f}$" % x


@ticker.FuncFormatter
def error_formatter(x, pos):
    return "%.1f×" % x


def forest_to_tree_nodes(G):
    """
    A forest is a set of unconnected trees. This function returns a list of sets, where each set
    contains the nodes of a single tree.
    """
    connected_components = list(nx.connected_components(G.to_undirected()))

    component_sizes = []
    for comp in connected_components: 
        component_sizes.append(len(comp)) 

    logging.info("Found {} clusters. {} clusters containt a single outlier, out of {} total points".format(
        len(connected_components), np.sum(np.array(component_sizes) == 1), np.sum(component_sizes)))

    return connected_components


def HDBSCAN_to_DBSCAN(G, original_nodes, epsilon):
    """
    Since running DBSCAN is slow, we reuse the existing minimum spanning tree provided from 
    HDBSCAN and remove any edges that are longer than epsilon. Whatever disjoint graphs are
    left form the clusters DBSCAN would have returned.

    Assumes that the nodes in the graph are labeled from 0 to the number of nodes - 1
    """
    G = copy.deepcopy(G)

    logging.info("Running DBSCAN with eps={} on the HDBSCAN minimum spanning tree".format(epsilon))

    # First, delete any edges that with 1 / weights larger than epsilon
    for edge in list(G.edges):
        if 1 / G[edge[0]][edge[1]]['weight'] > epsilon:
            G.remove_edge(*edge)

    clusters = forest_to_tree_nodes(G)

    # Since the graph contains some branch nodes, and not just leaf nodes, remove those branch nodes
    branch_nodes = range(original_nodes + 1, np.max(G.nodes()) + 1)
    for idx in range(len(clusters)):
        clusters[idx] = clusters[idx].difference(branch_nodes)

    # This will leave some empty clusters. Let's remove them
    clusters = [c for c in clusters if len(c) > 0]

    return clusters


def plot_parallel_coordinates(gs, df, dpi, name=""):
    """
    For the provided axes and the cluster, plots the basic information about the cluster in them.
    """
    split_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs, hspace=0.8)
    log_ax  = plt.subplot(split_gs[0])  # noqa: F821
    perc_ax = plt.subplot(split_gs[1])  # noqa: F821

    log_columns = ["POSIX_RAW_agg_perf_by_slowest", "POSIX_RAW_total_bytes", "RAW_runtime", "POSIX_RAW_total_accesses", "RAW_nprocs", "POSIX_RAW_total_files", "apps_short"]
    perc_columns = ["POSIX_BYTES_READ_PERC", 'POSIX_read_only_bytes_perc', 'POSIX_read_write_bytes_perc', 'POSIX_write_only_bytes_perc', "POSIX_unique_files_perc", "apps_short"]

    # Preprocess the log table data
    log_data = df[log_columns].copy()
    log_data.POSIX_RAW_total_bytes /= 1024 ** 3  # Convert to gigabytes 
    log_data.POSIX_RAW_total_accesses /= 1024    # Convert to kilo-transactions
    log_data.rename(columns={"POSIX_RAW_agg_perf_by_slowest": "Throughput\n[MB/s]", "POSIX_RAW_total_accesses": "R/W accesses\n(in 1000s)", "RAW_runtime": "runtime [s]", "RAW_nprocs": "App size\n(nprocs)", "POSIX_RAW_total_bytes": "Volume [GB]",   "POSIX_RAW_total_files": "files\n(count)"}, inplace=True)

    # Preprocess the percentage table data
    perc_data = df[perc_columns].copy()
    perc_data.rename(columns={"POSIX_BYTES_READ_PERC": "Read ratio\n(by volume)", 'POSIX_read_only_bytes_perc': "RO files\n(by file #)", 'POSIX_read_write_bytes_perc': "R/W files\n(by file #)", 'POSIX_write_only_bytes_perc': "WO files\n(by file #)", "POSIX_unique_files_perc": "Unique files\n(by file #)"}, inplace=True)
    
    # Title
    log_ax.set_title("Cluster {}:\n {} jobs".format(name, np.sum(df.shape[0])))

    # First, plot the logarithmic plot
    parallel_coordinates(log_data.sample(n=int(log_data.shape[0] / 10)), "apps_short", ax=log_ax, sort_labels=True, alpha=0.1)
    # for lh in log_ax.get_legend().legendHandles: lh.set_alpha(1)  # noqa: E701
    log_ax.get_legend().remove()
    log_ax.set_yscale('log')
    log_ax.grid(True)
    # log_ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    log_ax.set_xticklabels(log_ax.get_xticklabels(), rotation=37, ha="right", rotation_mode="anchor")
    log_ax.set_ylim(10**0, 10**6)

    # Then the percentage plot
    parallel_coordinates(perc_data.sample(n=int(perc_data.shape[0] / 10)), "apps_short", ax=perc_ax, sort_labels=True, alpha=0.1)
    # for lh in perc_ax.get_legend().legendHandles: lh.set_alpha(1)  # noqa: E701
    perc_ax.get_legend().remove()
    perc_ax.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    perc_ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    perc_ax.set_xticklabels(perc_ax.get_xticklabels(), rotation=37, ha="right", rotation_mode="anchor")

    # Shift labels 
    dx = 0; dy = 5/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, dpi)

    # apply offset transform to all x ticklabels.
    for label in log_ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    for label in perc_ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


def train_and_plot_errors(box_ax, shap_ax, corr_ax, scatter_axes, X, y, feature_num):
    """
    Splits the dataset into a training and test set, and trains an XGBoost model.
    Plots the test set errors in a box plot.
    Optionally plots a SHAP summary plot.
    """
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

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    xgb_model = xgb.XGBRegressor(obj=huber_approx_obj)
    lin_model = sklearn.linear_model.SGDRegressor(loss='huber')
    dmy_model = sklearn.dummy.DummyRegressor("median")
    xgb_model.fit(X_train, y_train, eval_metric=huber_approx_obj)
    lin_model.fit(X_train, y_train)
    dmy_model.fit(X_train, y_train)

    xgb_error = 10**np.abs(y_test - xgb_model.predict(X_test))
    lin_error = 10**np.abs(y_test - lin_model.predict(X_test))
    dmy_error = 10**np.abs(y_test - dmy_model.predict(X_test))

    df = pd.DataFrame.from_dict({"Error": np.concatenate([dmy_error, lin_error, xgb_error]), 
                                 "Classifier": ["Constant"]*len(dmy_error) + ["Linear"]*len(lin_error) + ["XGB"]*len(xgb_error)})
    #
    # Plotting error boxplots 
    #
    if box_ax is not None:
        sns.boxplot(data=df, y="Error", x="Classifier", ax=box_ax)
        box_ax.set_yscale('log')
        box_ax.set_ylabel('Relative error')
        box_ax.yaxis.set_major_formatter(error_formatter)

    #
    # Plotting SHAP summary plot
    #
    explainer = shap.TreeExplainer(xgb_model, shap.sample(X_train, 100))
    shap_values = explainer.shap_values(X_test)
    feature_names = list(map(feature_name_mapping.mapping.get, X.columns))
    if shap_ax is not None:
        shap_ax.plot([], []) # SHAP just uses whatever axis was used last. We need to print some
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=feature_num, color_bar=False)

    #
    # Plotting correlation matrix between most important features according to SHAP and any high-corr. features
    #
    most_important_features = list(reversed(X.columns[np.argsort(np.abs(shap_values).mean(0))]))[:feature_num]
    if corr_ax is not None:
        corr = (X.corr() - np.eye(X.shape[1]))[most_important_features] 
        second_axis_features = np.abs(corr.mean(1)).nlargest(feature_num).index.values
        matrix = corr[most_important_features].T[second_axis_features]
        corr_ax.matshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        corr_ax.xaxis.set_ticks_position('bottom')
        corr_ax.set_xticks(range(feature_num))
        corr_ax.set_yticks(range(feature_num))
        corr_ax.set_xticklabels(matrix.columns, rotation=30, ha="right") 
        corr_ax.set_yticklabels(matrix.index.values)  
        plt.colorbar()

    #
    # Plotting scatter plots with feature pairs taken as 1-2, 1-3, 1-4
    # If data volume is in the top features, use that on the X axis 
    #
    for idx in range(len(most_important_features)):
        if 'POSIX_LOG10_total_bytes' in most_important_features[idx]:
            most_important_features[0], most_important_features[idx] = most_important_features[idx], most_important_features[0] 

    if scatter_axes is not None: 
        for idx, ax in enumerate(scatter_axes): 
            points = ax.scatter(X[most_important_features[0]], X[most_important_features[idx+1]], c=y, cmap="jet", lw=0, alpha=0.4, s=4**2)
            cbar = plt.colorbar(points, ax=ax)
            # Labels 
            ax.set_xlabel(feature_name_mapping.mapping[most_important_features[0]])
            ax.set_ylabel(feature_name_mapping.mapping[most_important_features[idx+1]])
            cbar.set_label("I/O throughput [MB/s]")
            # X axis tick formatter
            if "perc" in most_important_features[0].lower():
                ax.xaxis.set_major_formatter(perc_formatter)
            elif "LOG10" in most_important_features[0]:
                ax.xaxis.set_major_formatter(log_formatter)
            # Y axis tick formatter
            if "perc" in most_important_features[idx+1].lower():
                ax.yaxis.set_major_formatter(perc_formatter)
            elif "LOG10" in most_important_features[idx+1]:
                ax.yaxis.set_major_formatter(log_formatter)
            # colobar ticks
            cbar.set_ticks(cbar.ax.get_yticks())
            cbar.set_ticklabels(["$10^{" + str(int(t)) + "}$" % t for t in cbar.ax.get_yticks()])
            # change background to make points more visible
            ax.set_facecolor("#5c5c5c")


def main(select_clusters, cluster_num, feature_num, scatter_num, epsilon, parallel_plots, box_plots, shap_plots, corr_plots):
    """
    """
    df, clusterer = dataset.default_dataset(paths=["../data/anonimized_io.csv"])
    # We remove a lot of the columns because SHAP sometimes picks them up. 
    input_columns = set([c for c in df.columns if 'perc' in c.lower() or 'LOG10' in c]).difference(["POSIX_LOG10_agg_perf_by_slowest", "LOG10_runtime", 
        "POSIX_LOG10_SEEKS", "POSIX_LOG10_MODE", "POSIX_LOG10_STATS", 'POSIX_ACCESS1_COUNT_PERC', 'POSIX_ACCESS2_COUNT_PERC', 'POSIX_ACCESS3_COUNT_PERC', 'POSIX_ACCESS4_COUNT_PERC'])

    # 
    # Since almost any epsilon is going to give us more clusters than we want (mostly outliers), here we refine them to TOP_CLUSTERS
    #
    if select_clusters is None: 
        clusters = HDBSCAN_to_DBSCAN(clusterer.condensed_tree_.to_networkx(), df.shape[0], epsilon=epsilon)
        top_indexes = reversed(np.argsort([len(x) for x in clusters])[-cluster_num:])
        top_clusters = [clusters[idx] for idx in top_indexes]
    # Given indexes of nodes in the condensed tree, find reachable leaves 
    else:
        def get_leaves(G): 
            return {x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1}

        G = clusterer.condensed_tree_.to_networkx()
        top_clusters = [get_leaves(nx.dfs_tree(G, c)) for c in select_clusters]

    #
    # Plotting setup
    #
    fig = plt.figure()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.6)
    rows =              int(parallel_plots)    + int(box_plots)    + int(shap_plots)    + int(corr_plots)     + int(scatter_num)
    height_ratios = [0.8] * parallel_plots + [0.3] * box_plots + [0.5] * shap_plots + [0.4] * corr_plots + [0.25] * scatter_num # Hardcoded for now 
    gs = fig.add_gridspec(rows, len(top_clusters), height_ratios=height_ratios, hspace=0.55)
    next_row = 0

    #
    # Plot cluster metadata in the next two rows
    #
    if parallel_plots:
        logging.info("Plotting parallel plots")
        for c_idx, cluster in enumerate(top_clusters): 
            plot_parallel_coordinates(gs[next_row, c_idx], df.iloc[list(cluster)], dpi=fig.dpi_scale_trans, name=str(c_idx))

        next_row += 1


    #
    # Next, let's fit linear models and predict POSIX_agg_perf_by_slowest
    # We will plot a boxplot of the test set points 
    #
    if box_plots:
        logging.info("Training models")
        for c_idx, cluster in enumerate(top_clusters): 
            box_ax       = fig.add_subplot(gs[next_row, c_idx], sharey=box_ax if c_idx != 0 else None)   # noqa: F821
            scatter_axes = [fig.add_subplot(gs[next_row + 2 + s, c_idx]) for s in range(scatter_num)] if scatter_num > 0 else None
            corr_ax      = fig.add_subplot(gs[next_row + 2, c_idx]) if corr_plots else None
            shap_ax      = fig.add_subplot(gs[next_row + 1, c_idx]) if shap_plots else None
            train_and_plot_errors(box_ax, shap_ax, corr_ax, scatter_axes, df.iloc[list(cluster)][input_columns], df.iloc[list(cluster)].POSIX_LOG10_agg_perf_by_slowest, feature_num)

        next_row += 3

    plt.show()
    

if __name__ == "__main__":
    random.seed(0xdeadbeef)
    np.random.seed(0xdeadbeef)

    parser = argparse.ArgumentParser()
    # If the user provides a list of cluster indices (e.g., from the HDBSCAN dendogram, then we shouldn't use -c and -e
    parser.add_argument('--select_clusters', nargs='+', type=int)
    parser.add_argument("-c", "--cluster_num", default=5, type=int)
    parser.add_argument("-e", "--epsilon", default=1., type=float)
    # Plotting arguments
    parser.add_argument("-f", "--feature_num", default=4, type=int)
    parser.add_argument("-s", "--scatter_num", default=2, type=int)
    parser.add_argument("--no-parallel-plots", action='store_false', dest='parallel_plots')
    parser.add_argument("--no-box-plots", action='store_false', dest='box_plots')
    parser.add_argument("--no-shap-plots", action='store_false', dest='shap_plots')
    parser.add_argument("--corr-plots", action='store_true', dest='corr_plots')

    args = parser.parse_args()
    main(**vars(args))



