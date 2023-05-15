from src.graph import Graph
from src.node import Node

import numpy as np 


def update_pagerank_node(node: Node, damping_factor: float, n: int):
    """Update the pagerank of the node.

    Args:
        damping_factor (float): Damping factor, 1 - probability of randomly walking out the links.
        n (int): Number of nodes in the graph."""
    
    pagerank_sum = sum((node.pagerank / len(node.children)) for node in node.parents)
    random_jumping = (1 - damping_factor) / n
    node.pagerank = random_jumping + damping_factor * pagerank_sum

def update_pagerank_one_iter(graph: Graph, damping_factor: float):
    if damping_factor < 0 or damping_factor > 1:
        raise ValueError("d must be between 0 and 1")
    
    node_list = graph.nodes
    n = len(node_list)
    for node in node_list:
        update_pagerank_node(node, damping_factor, n)

    graph.normalize_pagerank()

def compute_pagerank(graph: Graph, damping_factor: float, max_iter: int=100):
    for _ in range(max_iter):
        update_pagerank_one_iter(graph, damping_factor)
        prev_pagerank = graph.get_pageranks()

        if np.abs(prev_pagerank - graph.get_pageranks()).max() < 1e-6:
            break