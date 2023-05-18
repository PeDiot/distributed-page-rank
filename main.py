from src.pagerank.pagerank_numpy import compute_pagerank_numpy
from src.pagerank.pagerank_cython import compute_pagerank_cython
from src.pagerank.pagerank_base import compute_pagerank_base

from src.experiment import Experiment

from src.utils import make_computation_time_table, make_mse_table
from src.plots import plot_computation_times, plot_mse
from src.backup import load_config, to_json, from_json

import rich
import argparse


def main(type: str, hyperparam: str): 
    """Run an experiment or evaluate the results of an experiment.
    
    Args:
        type (str): The type of the experiment. Either 'exp' or 'eval'.
        hyperparam (str): The hyperparameter to evaluate. Either 'n_nodes', 'min_conn_per_node' or 'max_iter'."""
    
    cfg = load_config("config.yaml") 
    n_nodes, min_conn_per_node, max_iter = cfg["n_nodes"], cfg["min_conn_per_node"], cfg["max_iter"]

    if type == "exp":
        exp = Experiment(n_nodes, min_conn_per_node, max_iter)
        rich.print(exp)

        exp.init_graph_memory()

        results_base = exp.run(
            pagerank_method=compute_pagerank_base, 
            damping_factor=cfg["damping_factor"], 
            n_repeat=cfg["n_repeat"])
        to_json(results_base, "backup", f"base_{hyperparam}.json")

        results_numpy = exp.run(
            pagerank_method=compute_pagerank_numpy, 
            damping_factor=cfg["damping_factor"], 
            n_repeat=cfg["n_repeat"])
        to_json(results_numpy, "backup", f"numpy_{hyperparam}.json")

        results_cython = exp.run(
          pagerank_method=compute_pagerank_cython, 
          damping_factor=cfg["damping_factor"], 
          n_repeat=cfg["n_repeat"], 
          num_threads=cfg["num_threads"])
        to_json(results_cython, "backup", f"cython_{hyperparam}.json")

    elif type == "eval":
        base_results = from_json("backup", f"base_{hyperparam}.json")
        numpy_results = from_json("backup", f"numpy_{hyperparam}.json")
        cython_results = from_json("backup", f"cython_{hyperparam}.json")
        gpu_results = from_json("backup", f"gpu_{hyperparam}.json")

        base_time_table = make_computation_time_table(base_results, "base")
        numpy_time_table = make_computation_time_table(numpy_results, "numpy")
        cython_time_table = make_computation_time_table(cython_results, "cython")
        gpu_time_table = make_computation_time_table(gpu_results, "gpu")

        mse_table = make_mse_table(
            {
                "base": base_results,
                "numpy": numpy_results, 
                "cython": cython_results, 
                "gpu": gpu_results
            }
        )
        rich.print(mse_table)
        
        if hyperparam == "n_nodes":
            plot_title = f"{min_conn_per_node=} | {max_iter=}"
        elif hyperparam == "min_conn_per_node":
            plot_title = f"{n_nodes=} | {max_iter=}"
        elif hyperparam == "max_iter":
            plot_title = f"{n_nodes=} | {min_conn_per_node=}"
        else:
            raise ValueError(f"Invalid hyperparameter {hyperparam}. Must be either 'n_nodes', 'min_conn_per_node' or 'max_iter'.")
        
        plot_mse(df=mse_table, x_var="n_nodes", title=plot_title, file_name=f"{hyperparam}.png")

        plot_computation_times(
            results=[
                base_time_table, 
                numpy_time_table, 
                cython_time_table,
                gpu_time_table
            ], 
            x_var="n_nodes", 
            title=plot_title, 
            file_name=f"{hyperparam}.png")
        
    else:
        raise ValueError("Invalid type. Must be either 'exp' or 'eval'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an experiment or evaluate the results of an experiment.")
    parser.add_argument("-t", "--type", help="The type of the experiment. Either 'exp' or 'eval'.", default="eval")
    parser.add_argument("-hp", "--hyperparam", help="The hyperparameter to evaluate. Either 'n_nodes', 'min_conn_per_node' or 'max_iter'.", default="n_nodes")
    args = parser.parse_args()
    main(args.type, args.hyperparam)