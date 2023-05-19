from typing import List, Union, Callable

import numpy as np 
from src.graph import generate_random_adjacency_matrix
from src.utils import measure_time


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
    
    def init_graph_memory(self): 
        self.graphs = {}

    def run(self, pagerank_method: Callable, n_repeat: int=50, *args, **kwargs) -> List: 
        """Runs the experiment for different pagerank computation methods.

        Args:
            pagerank_methods (Callable): The pagerank computation method to use.
            n_repeat (int): The number of times to repeat the experiment. Defaults to 50.

        Returns:
            List: A list containing the time measurements and the pangerank values."""
        
        results = []

        method_name = pagerank_method.__name__.split("_")[-1]

        for attrs in self.attrs_list:
            n_nodes = attrs["n_nodes"]
            min_conn_per_node = attrs["min_conn_per_node_prop"]
            max_iter = attrs["max_iter"]
            
            info = {
                "n_nodes": n_nodes,
                "min_conn_per_node": min_conn_per_node,
                "max_iter": max_iter,
            }
            results.append(info)

            key = f"{n_nodes}_{min_conn_per_node}"

            if not key in self.graphs:
                graph_coo = generate_random_adjacency_matrix(n_nodes=n_nodes, min_conn_per_node=min_conn_per_node)
                self.graphs[key] = graph_coo            

            desc = f"{method_name} | {n_nodes=}  | {min_conn_per_node=} | {max_iter=}"
            time, values = measure_time(
                pagerank_method,
                graph_coo=self.graphs[key],
                max_iter=max_iter,
                return_results=True,
                desc=desc, 
                n_repeat=n_repeat,
                *args,
                **kwargs)
            
            if method_name == "cython":
                values = np.asarray(values)

            results[-1][f"time_{method_name}"] = time
            results[-1][f"pagerank_{method_name}"] = values.tolist()
            
        return results