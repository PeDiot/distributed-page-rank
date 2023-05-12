from src.utils import compute_time, generate_random_edges
from src.graph import Graph
from src.pagerank import compute_pagerank

import rich
import argparse

def main(n_edges: int, d: float, n_iter: int, save: bool): 
    edges = generate_random_edges(n_edges)
    graph = Graph.from_edges(edges)

    rich.print(f"{n_edges} random edges in the graph.")
    result = compute_time(compute_pagerank, graph=graph, d=d, n_iter=100) 
    rich.print(f"Execution time summary: {result}")

    if save:
        figure_file=f"figs/graph_{n_edges}.png"
        pageranks = graph.get_pageranks(save_path=f"backup/pageranks_{n_edges}.json")
    else: 
        figure_file = None
        pageranks= graph.get_pageranks()

    rich.print(f"Updated PR values with {d=} after {n_iter} iterations: {pageranks}") 
    graph.visualize(figure_file=figure_file)

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_edges", default=10)
parser.add_argument("-d", "--d", default=0.85)
parser.add_argument("-n_iter", "--n_iter", default=100)
parser.add_argument("-save", "--save", default=False)
args = parser.parse_args()

main(int(args.n_edges), float(args.d), int(args.n_iter), args.save)