"""
Produces Figure 1 from our paper. You may not get the same image because we had to cut down the data to only jobs > 100MB of I/O.
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# Import root module
sys.path.insert(0, '../')
import dataset

sns.set


def main():
    df, _ = dataset.default_dataset(paths=["../data/anonimized_io.csv"])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    options = {"fontname": "Times New Roman", "fontsize": 18}
    options_small = {"fontname": "Times New Roman", "fontsize": 16}
    plt.subplots_adjust(wspace=0.05)

    # Plotting the first figure. We multiply by three because the old code used log2 and not log10 features
    hexbin = ax1.hexbin(x=df.POSIX_LOG10_total_bytes*3, y=df.POSIX_LOG10_agg_perf_by_slowest*3, bins="log", linewidths=0.1, cmap="bone_r", gridsize=64, vmax=3*10**4)

    ax1.set_xticks([10, 20, 30, 40, 50])
    ax1.set_xticklabels(["KiB", "MiB", "GiB", "TiB", "PiB"], **options_small)
    ax1.set_yticks([-10, 0, 10, 20])
    ax1.set_yticklabels(["KiB / s", "MiB / s", "GiB / s", "TiB / s", "PiB /s"], **options_small)
    ax1.set_ylabel("I/O Throughput", **options)
    ax1.set_xlabel("I/O Volume", **options)
    ax1.grid(alpha=0.3)

    # Plotting the 2nd figure
    hexbin = ax2.hexbin(x=df.LOG10_nprocs*3, y=df.POSIX_LOG10_agg_perf_by_slowest*3, bins="log", linewidths=0.1, cmap="bone_r", gridsize=64, vmax=3*10**4)

    ax2.set_xticks([0, 4, 8, 12, 16])
    ax2.set_xticklabels([1, 16, 256, 4096, 65536], **options_small)
    ax2.set_yticks([-10, 0, 10, 20])
    ax2.set_yticklabels(["KiB / s", "MiB / s", "GiB / s", "TiB / s", "PiB /s"], **options_small)
    ax2.set_xlabel("Number of processes", **options)
    ax2.grid(alpha=0.3)

    cbar = plt.colorbar(hexbin, ax=ax2)
    cbar.set_label("Number of jobs", **options)

    plt.show()



if __name__ == "__main__": 
    main()
