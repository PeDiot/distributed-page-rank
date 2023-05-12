from .graph import Graph


def update_pagerank_one_iter(graph: Graph, d: float):
    if d < 0 or d > 1:
        raise ValueError("d must be between 0 and 1")
    
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))

    graph.normalize_pagerank()

def compute_pagerank(graph: Graph, d: float, n_iter: int=100):
    for _ in range(n_iter):
        update_pagerank_one_iter(graph, d)