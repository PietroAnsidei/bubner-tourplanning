"""Test results and output plots."""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.setup import setup_params

if __name__ == "__main__":

    # Import parameters
    params = setup_params()

    # Import results & drop duplicates, remove missing values
    tests = (
        pd.read_csv(
            f"{params['data_folder']}/{params['output_test_file']}", index_col=0
        )
        .drop_duplicates()
        .sort_values(["strategy", "local_search"])
        .replace(0, np.nan)
    )

    # Create figure
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))

    # Visualize performances of different search strategies and local optimization
    for idx_var, var in enumerate(
        ["sum_total_distances", "sum_total_time", "duration"]
    ):

        for idx_split, split in enumerate(["strategy", "local_search"]):

            # Create boxplot
            sns.boxplot(data=tests, x=split, y=var, ax=ax[idx_var, idx_split])

            # Axis visualization (only the last)
            if idx_var < 2:
                ax[idx_var, idx_split].set_xticklabels([])
                ax[idx_var, idx_split].set(xlabel=None)
            else:
                ax[idx_var, idx_split].xaxis.set_tick_params(labelsize=7, rotation=45)
            if idx_split == 1:
                ax[idx_var, idx_split].set_yticklabels([])
                ax[idx_var, idx_split].set(ylabel=None)

            ax[idx_var, idx_split].grid()

    # Store plot
    plt.tight_layout()
    plt.savefig(f"{params['data_folder']}/{params['output_test_plot']}")
