from typing import Union
from .graph import Graph
from tqdm import tqdm


def update_pagerank_one_iter(graph: Graph, d: float):
    if d < 0 or d > 1:
        raise ValueError("d must be between 0 and 1")
    
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))

    graph.normalize_pagerank()

def compute_pagerank(graph: Graph, d: float, n_iter: int=100):
    loop = tqdm(range(n_iter))
    loop.set_description(f"Computing PageRank with damping_factor={d}")
    
    for i in loop:
        update_pagerank_one_iter(graph, d)