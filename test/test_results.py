"""Test results and output plots."""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # Import results & drop duplicates
    tests = (
        pd.read_csv(
            f"{params['data_folder']}/{params['output_test_file']}", index_col=0
        )
        .drop_duplicates()
        .sort_values(["strategy", "local_search"])
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for idx_var, var in enumerate(["sum_total_distances", "sum_total_time"]):
        for idx_split, split in enumerate(["strategy", "local_search"]):
            sns.boxplot(data=tests, x=split, y=var, ax=ax[idx_var, idx_split])
            if idx_var == 0:
                ax[idx_var, idx_split].set_xticks([])
            else:
                ax[idx_var, idx_split].xaxis.set_tick_params(labelsize=7, rotation=45)
            ax[idx_var, idx_split].grid()

    plt.savefig(f"{params['data_folder']}/{params['output_test_plot']}")
