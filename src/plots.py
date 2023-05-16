from typing import Dict, List, Tuple, Optional

import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import dict_to_dataframe


def plot_computation_times(
    results: List[Dict], 
    x_var: str, 
    title: str, 
    save_path: Optional[str]=None, 
    fig_dims: Tuple=(18, 14)
) -> None:
    """Plots the computation times of different pagerank computation methods.

    Args:
        results: A list of dictionaries containing the results of the experiment.
        x_var: The variable to plot on the x-axis (n_nodes, min_conn_per_node, or max_iter)
        title: The title of the plot.
        save_path: The path to save the plot to. If None, the plot is not saved.
        fig_dims: The dimensions of the plot."""
    
    df = dict_to_dataframe(results)
    y_vars = [
        "min_time", 
        "max_time", 
        "mean_time", 
        "median_time",
        "std_time"
    ]
    
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=fig_dims)
    fig.suptitle(title, fontsize=12)

    for y_var, ax in zip(y_vars, axes.flatten()):
        sns.lineplot(x=x_var, y=y_var, hue="method", data=df, ax=ax)
    
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()