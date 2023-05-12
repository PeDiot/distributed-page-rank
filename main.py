"""Description. Compute PageRank values and visualize the graph.

Example: `python main.py -n 1 -save True`

Initial PR values: [1. 1. 1. 1. 1. 1.]
Computing PageRank with damping_factor=0.15: 100%|███████████████████████████████████████| 100/100 [00:00<00:00, 37858.15it/s]
Updated PR values with d=0.15 after 100 iterations: [0.061 0.112 0.156 0.193 0.225 0.252]"""

from src.utils import file_to_edges
from src.graph import Graph
from src.pagerank import compute_pagerank

import rich
import argparse

def main(graph_number: int, d: float, n_iter: int, save: bool): 
    edges = file_to_edges(f"datasets/graph_{graph_number}.txt")
    graph = Graph.from_edges(edges)

    rich.print(f"Initial PR values: {graph.get_pagerank_list()}")

    compute_pagerank(graph, 0.15, 100)
    rich.print(f"Updated PR values with {d=} after {n_iter} iterations: {graph.get_pagerank_list()}") 

    if save:
        graph.visualize(figure_file=f"figs/graph_{graph_number}.png")
    else: 
        graph.visualize()

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--graph_number", default=1)
parser.add_argument("-d", "--d", default=0.15)
parser.add_argument("-n_iter", "--n_iter", default=100)
parser.add_argument("-save", "--save", default=False)
args = parser.parse_args()

main(int(args.graph_number), float(args.d), int(args.n_iter), args.save)