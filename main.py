from src.utils import file_to_edges
from src.graph import Graph
from src.pagerank import compute_pagerank

import rich

edges = file_to_edges("datasets/graph_1.txt")
graph = Graph.from_edges(edges)

rich.print(graph.display()) 
rich.print(graph.get_pagerank_list())

compute_pagerank(graph, 0.85, 10)
rich.print(graph.get_pagerank_list()) 