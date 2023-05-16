from typing import Optional, List, Union, Dict, Callable
from src.graph import generate_random_adjacency_matrix
from src.utils import measure_time, compute_mse


class Experiment: 
    """A class for running experiments on pagerank computation over different graph configurations.

    Attributes:
        n_nodes (int): The number of nodes in the graph.
        min_conn_per_node_prop (float): The proportion of nodes each node is connected to.
        max_iter (int): The maximum number of iterations to run the pagerank algorithm for."""

    def __init__(
        self, 
        n_nodes: Union[List[int], int], 
        min_conn_per_node: Union[float, List[float]],  
        max_iter: Union[int, List[int]]
    ): 
        if isinstance(n_nodes, List):
            if isinstance(min_conn_per_node, List) or isinstance(max_iter, List):
                raise ValueError("Only one of node_sizes, min_conn_per_node_prop, and max_iter can be a list")
            if min_conn_per_node <= 0 or min_conn_per_node >= 1:
                raise ValueError("min_conn_per_node_prop must be between 0 and 1")
            self.attrs_list = [
                {
                    "n_nodes": n,
                    "min_conn_per_node_prop": min_conn_per_node,
                    "max_iter": max_iter
                }
                for n in n_nodes
            ]

        elif isinstance(min_conn_per_node, List):
            if isinstance(n_nodes, List) or isinstance(max_iter, List):
                raise ValueError("Only one of node_sizes, min_conn_per_node_prop, and max_iter can be a list")
            self.attrs_list = [
                {
                    "n_nodes": n_nodes,
                    "min_conn_per_node_prop": p,
                    "max_iter": max_iter
                }
                for p in min_conn_per_node
            ]

        elif isinstance(max_iter, List):
            if isinstance(n_nodes, List) or isinstance(min_conn_per_node, List):
                raise ValueError("Only one of node_sizes, min_conn_per_node_prop, and max_iter can be a list")
            if min_conn_per_node <= 0 or min_conn_per_node >= 1:
                raise ValueError("min_conn_per_node_prop must be between 0 and 1")
            self.attrs_list = [
                {
                    "n_nodes": n_nodes,
                    "min_conn_per_node_prop": min_conn_per_node,
                    "max_iter": max_it
                }
                for max_it in max_iter
            ]

        else: 
            if min_conn_per_node <= 0 or min_conn_per_node >= 1:
                raise ValueError("min_conn_per_node_prop must be between 0 and 1")
            self.attrs_list = [
                {
                    "n_nodes": n_nodes,
                    "min_conn_per_node_prop": min_conn_per_node,
                    "max_iter": max_iter
                }
            ]

        self.n_nodes, self.min_conn_per_node, self.max_iter = n_nodes, min_conn_per_node, max_iter

    def __repr__(self) -> str:
        return f"Experiment(n_nodes={self.n_nodes}, min_conn_per_node={self.min_conn_per_node}, max_iter={self.max_iter})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def run(self, pagerank_methods: Dict[str, Callable], n_repeat: int, *args, **kwargs) -> List: 
        """Runs the experiment for different pagerank computation methods.

        Args:
            pagerank_methods: A dictionary of pagerank computation methods to use.
            n_repeat: The number of times to repeat the experiment.

        Returns:
            A list of dictionaries containing the results of the experiment."""
        
        results = []

        for attrs in self.attrs_list:
            n_nodes = attrs["n_nodes"]
            min_conn_per_node = attrs["min_conn_per_node_prop"]
            max_iter = attrs["max_iter"]

            results.append(
                {
                    "n_nodes": n_nodes,
                    "min_conn_per_node": min_conn_per_node,
                    "max_iter": max_iter,
                }
            )

            pageranks = {}

            graph_coo = generate_random_adjacency_matrix(n_nodes=n_nodes, min_conn_per_node=min_conn_per_node)

            for method, fun in pagerank_methods.items():
                desc = f"{method} | {n_nodes=}  | {min_conn_per_node=} | {max_iter=}"
                time, values = measure_time(
                    fun,
                    graph_coo=graph_coo,
                    max_iter=max_iter,
                    return_results=True,
                    desc=desc, 
                    *args,
                    **kwargs)

                pageranks[method] = values
                results[-1][f"time_{method}"] = time

            if len(pageranks) > 1:
                try: 
                    mse_cython = compute_mse(pageranks["numpy"], pageranks["cython"])
                    results[-1]["mse_cython"] = mse_cython
                except KeyError:
                    pass
            
        return results