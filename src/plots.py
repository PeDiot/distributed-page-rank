from typing import Union, List, Tuple, Optional

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_computation_times(
    results: Union[pd.DataFrame, List[pd.DataFrame]], 
    x_var: str, 
    title: Optional[str]=None, 
    file_name: Optional[str]=None, 
    fig_dims: Tuple=(18, 14), 
    show_plot: bool=False
) -> None:
    """Plots the computation times of different pagerank computation methods.

    Args:
        results: A list of dictionaries containing the results of the experiment.
        x_var: The variable to plot on the x-axis (n_nodes, min_conn_per_node, or max_iter)
        title: The title of the plot.
        file_name: The file name to save the plot to. If None, the plot is not saved.
        fig_dims: The dimensions of the plot.
        show_plot: Whether to show the plot or not."""
    
    y_vars = [
        "min_time", 
        "max_time", 
        "mean_time", 
        "median_time",
        "std_time"
    ]

    if not isinstance(results, pd.DataFrame):
        df = pd.concat(results, axis=0).reset_index(drop=True)
    
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=fig_dims)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    for y_var, ax in zip(y_vars, axes.flatten()):
        sns.lineplot(x=x_var, y=y_var, hue="method", data=df, ax=ax)

    fig.delaxes(axes.flatten()[-1])
    
    if file_name is not None:
        save_path = f"figs/time"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, file_name)
        plt.savefig(save_path)

    if show_plot:
        plt.show()

def plot_mse(
    df: pd.DataFrame, 
    x_var: str, 
    title: Optional[str]=None, 
    file_name: Optional[str]=None, 
    fig_dims: Tuple=(9, 4), 
    show_plot: bool=False
) -> None: 
    """Plots the MSE of different pagerank computation methods.

    Args:
        df: A pandas DataFrame containing the results of the experiment.
        x_var: The variable to plot on the x-axis (n_nodes, min_conn_per_node, or max_iter)
        title: The title of the plot.
        file_name: The file name to save the plot to. If None, the plot is not saved.
        fig_dims: The dimensions of the plot.
        show_plot: Whether to show the plot or not."""

    fig, ax = plt.subplots(figsize=fig_dims)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    sns.lineplot(x=x_var, y="mse", hue="method", data=df, ax=ax)

    if file_name is not None:
        save_path = f"figs/mse"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, file_name)
        plt.savefig(save_path)
        
    if show_plot:
        plt.show()