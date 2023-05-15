from typing import List, Optional, Dict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import time 
from tqdm import tqdm
import json

from src.graph import Graph


def file_to_edges(fname) -> List:
    """Returns a list of edges from txt file."""

    if not fname.endswith(".txt"):
        raise ValueError("File must be a .txt file")

    with open(fname) as f:
        lines = f.readlines()

    edges = [
        line.strip().split(",") for line in lines
    ]
    return edges


def compute_time(func, n_repeat: int=50, return_results: bool=False, *args, **kwargs):
    """Returns the time taken to run a function."""

    times = []
    loop = tqdm(range(n_repeat))

    for _ in loop:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
    
    res = {
        "min": min(times),
        "max": max(times),
        "mean": np.mean(times),
        "median": np.median(times), 
        "std": np.std(times)
    }

    if return_results:
        return res, result
    return res

def compute_mse(x: np.ndarray, y: np.ndarray) -> float:
    """Returns the mean squared error between two arrays."""

    if len(x) != len(y):
        raise ValueError("Arrays must be of same length")

    mse = np.sum((x - y) ** 2) / len(x)
    return mse

def visualize_graph(graph: Graph, figure_file: Optional[str] = None):
    """Visualize the graph using networkx and matplotlib."""

    G = nx.DiGraph()
    node_labels = {}

    for node in graph.nodes:
        G.add_node(node.name)
        node_labels[node.name] = f"{node.name} ({node.pagerank:.2f})"

        for child in node.children:
            G.add_edge(node.name, child.name)

    pageranks = [node.pagerank for node in graph.nodes]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=pageranks, cmap=plt.cm.Blues)

    if figure_file:
        plt.savefig(figure_file)
        
    plt.show()

def pageranks_to_dict(graph: Graph, save_path: Optional[str]=None) -> Dict:
    """Returns a dictionary of node names and pagerank values."""

    pageranks = {
        node.name: round(node.pagerank, 3) for node in graph.nodes
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(pageranks, f)

    return pageranks