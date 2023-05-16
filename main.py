from src.pagerank.pagerank_numpy import compute_pagerank_numpy
from src.pagerank.pagerank_cython import compute_pagerank_cython
from src.experiment import Experiment
from src.utils import make_computation_time_table, make_mse_table
from src.plots import plot_computation_times, plot_mse
from src.backup import load_config, to_json, from_json

import rich
import argparse


def main(type: str): 
    
    cfg = load_config("config.yaml") 
    n_nodes, min_conn_per_node, max_iter = cfg["n_nodes"], cfg["min_conn_per_node"], cfg["max_iter"]

    if type == "exp":
        exp = Experiment(n_nodes, min_conn_per_node, max_iter)
        rich.print(exp)

        exp.init_graph_memory()

        results_numpy = exp.run(pagerank_method=compute_pagerank_numpy, n_repeat=cfg["n_repeat"])
        rich.print(results_numpy)

        file_name = f"numpy_{n_nodes}_{min_conn_per_node}_{max_iter}"
        to_json(results_numpy, "backup", f"{file_name}.json")

        results_cython = exp.run(pagerank_method=compute_pagerank_cython, n_repeat=cfg["n_repeat"], num_threads=cfg["num_threads"])
        rich.print(results_cython)

        file_name = f"cython_{n_nodes}_{min_conn_per_node}_{max_iter}"
        to_json(results_cython, "backup", f"{file_name}.json")

    elif type == "eval":
        numpy_results = from_json("backup", f"numpy_{n_nodes}_{min_conn_per_node}_{max_iter}.json")
        cython_results = from_json("backup", f"cython_{n_nodes}_{min_conn_per_node}_{max_iter}.json")

        numpy_time_table = make_computation_time_table(numpy_results, "numpy")
        cython_time_table = make_computation_time_table(cython_results, "cython")

        plot_title = f"{n_nodes=} | {min_conn_per_node=} | {max_iter=}"
        figure_file = f"{n_nodes}_{min_conn_per_node}_{max_iter}.png"
        mse_table = make_mse_table({"numpy": numpy_results, "cython": cython_results})
        
        plot_mse(mse_table, "n_nodes", plot_title, f"figs/mse/{figure_file}")

        plot_computation_times(
            [numpy_time_table, cython_time_table], 
            "n_nodes", 
            plot_title, 
            f"figs/time/{figure_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an experiment or evaluate the results of an experiment.")
    parser.add_argument("-t", "--type", help="The type of the experiment. Either 'exp' or 'eval'.", default="eval")
    args = parser.parse_args()
    main(args.type)