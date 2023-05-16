from src.pagerank.pagerank_numpy import compute_pagerank_numpy
from src.pagerank.pagerank_cython import compute_pagerank_cython
from src.experiment import Experiment
from src.utils import load_config, to_json
from src.plots import plot_computation_times

import rich

def main(): 
    cfg = load_config("config.yaml") 

    pagerank_methods = {
        "numpy": compute_pagerank_numpy, 
        "cython": compute_pagerank_cython
    }

    n_nodes, min_conn_per_node, max_iter = cfg["n_nodes"], cfg["min_conn_per_node"], cfg["max_iter"]
    exp = Experiment(n_nodes, min_conn_per_node, max_iter)
    rich.print(exp)

    results = exp.run(pagerank_methods, n_repeat=cfg["n_repeat"])
    rich.print(results)

    file_name = f"results_{n_nodes}_{min_conn_per_node}_{max_iter}"
    to_json(results, "backup", f"{file_name}.json")

    title = f"Computation times for {n_nodes=}, {min_conn_per_node=}, {max_iter=}"
    plot_computation_times(results, "n_nodes", "Computation times per method", f"{file_name}.png")

if __name__ == "__main__":
    main()