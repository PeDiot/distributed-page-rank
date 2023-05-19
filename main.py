from src.pagerank.pagerank_numpy import compute_pagerank_numpy
from src.pagerank.pagerank_cython import compute_pagerank_cython
from src.pagerank.pagerank_base import compute_pagerank_base

from src.experiment import Experiment

from src.utils import make_computation_time_table, make_mse_table
from src.plots import plot_computation_times, plot_mse
from src.backup import load_config, to_json, from_json

from typing import List
import rich
import argparse


def main(
    type: str, 
    hyperparam: str, 
    pagerank_methods: List[str],
    log_y: bool=False
): 
    """Run an experiment or evaluate the results of an experiment.
    
    Args:
        type (str): The type of the experiment. Either 'exp' or 'eval'.
        hyperparam (str): The hyperparameter to evaluate. Either 'n_nodes', 'min_conn_per_node' or 'max_iter'.
        pagerank_methods (List[str]): The pagerank methods to evaluate. Possible values are 'base', 'numpy' or 'cython'.
        log_y (bool, optional): Whether to use a logarithmic y-axis or not. Defaults to False."""
    
    cfg = load_config("config.yaml") 
    n_nodes, min_conn_per_node, max_iter = cfg["n_nodes"], cfg["min_conn_per_node"], cfg["max_iter"]

    if type == "exp":
        exp = Experiment(n_nodes, min_conn_per_node, max_iter)
        rich.print(exp)

        exp.init_graph_memory()

        if "base" in pagerank_methods:
            results_base = exp.run(
                pagerank_method=compute_pagerank_base, 
                damping_factor=cfg["damping_factor"], 
                n_repeat=cfg["n_repeat"])
            to_json(results_base, "backup", f"base_{hyperparam}.json")

        if "numpy" in pagerank_methods:
            results_numpy = exp.run(
                pagerank_method=compute_pagerank_numpy, 
                damping_factor=cfg["damping_factor"], 
                n_repeat=cfg["n_repeat"])
            to_json(results_numpy, "backup", f"numpy_{hyperparam}.json")

        if "cython" in pagerank_methods:
            results_cython = exp.run(
                pagerank_method=compute_pagerank_cython, 
                damping_factor=cfg["damping_factor"], 
                n_repeat=cfg["n_repeat"], 
                num_threads=cfg["num_threads"])
            to_json(results_cython, "backup", f"cython_{hyperparam}.json")

    elif type == "eval":
        rich.print(f"Evaluating results for {hyperparam} | {pagerank_methods=}.")

        results_per_method = {}
        results = []

        if "base" in pagerank_methods:
            base_results = from_json("backup", f"base_{hyperparam}.json")
            base_time_table = make_computation_time_table(base_results, "base")
            results_per_method["base"] = base_results
            results.append(base_time_table)

        if "numpy" in pagerank_methods:
            numpy_results = from_json("backup", f"numpy_{hyperparam}.json")
            numpy_time_table = make_computation_time_table(numpy_results, "numpy")
            results_per_method["numpy"] = numpy_results
            results.append(numpy_time_table)

        if "cython" in pagerank_methods:
            cython_results = from_json("backup", f"cython_{hyperparam}.json")
            cython_time_table = make_computation_time_table(cython_results, "cython")
            results_per_method["cython"] = cython_results
            results.append(cython_time_table)

        if "gpu" in pagerank_methods:
            gpu_results = from_json("backup", f"gpu_{hyperparam}.json") 
            gpu_time_table = make_computation_time_table(gpu_results, "gpu")
            results_per_method["gpu"] = gpu_results
            results.append(gpu_time_table)

        figure_file_name = f"{hyperparam}_{pagerank_methods}.png"

        if "base" in pagerank_methods and len(pagerank_methods) > 1:      
            mse_table = make_mse_table(results_per_method) 
            rich.print(mse_table)
            plot_mse(df=mse_table, x_var=hyperparam, file_name=figure_file_name)

        if log_y:
            figure_file_name = f"log_{figure_file_name}"
        plot_computation_times(results, x_var=hyperparam, file_name=figure_file_name, log_y=log_y)
        
    else:
        raise ValueError("Invalid type. Must be either 'exp' or 'eval'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an experiment or evaluate the results of an experiment.")
    parser.add_argument("-t", "--type", help="The type of the experiment. Either 'exp' or 'eval'.", default="eval")
    parser.add_argument("-hp", "--hyperparam", help="The hyperparameter to evaluate. Either 'n_nodes', 'min_conn_per_node' or 'max_iter'.", default="n_nodes")
    parser.add_argument("-pm", "--pagerank_methods", help="The pagerank methods to evaluate.", default="base,numpy,cython,gpu")
    parser.add_argument("-ly", "--log_y", help="Whether to use a logarithmic y-axis or not.", type=bool, default=False)

    args = parser.parse_args()
    pagerank_methods = args.pagerank_methods.split(",")
    main(args.type, args.hyperparam, pagerank_methods, args.log_y)