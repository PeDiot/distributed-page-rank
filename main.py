from src.utils import (
    compute_time, 
    pageranks_to_dict, 
    visualize_graph, 
    compute_mse
) 
from src.graph import Graph, generate_random_adjacency_matrix, generate_random_edges
from src.pagerank.base import compute_pagerank
from src.pagerank.coo import compute_pagerank_from_coo
from src.pagerank.parallel import compute_pagerank_joblib

import rich
import argparse

def main(args): 
    if args.from_egdes: 
        n_edges = int(args.n_edges)
        edges = generate_random_edges(n_edges) 
        graph = Graph.from_edges(edges)
        rich.print(f"{args.n_edges} random edges in the graph.")
    else:
        n_nodes, min_conn_per_node = int(args.n_nodes), int(args.min_conn_per_node)
        coo = generate_random_adjacency_matrix(n_nodes, min_conn_per_node) 
        graph = Graph.from_adjacency_matrix(coo.toarray())
        rich.print(f"{args.n_nodes} random nodes in the graph with a minimum of {args.min_conn_per_node} connections per node.")

    damping_factor = float(args.damping_factor) 
    max_iter = int(args.max_iter)

    result_base = compute_time(compute_pagerank, graph=graph, damping_factor=args.damping_factor, max_iter=args.max_iter) 
    pageranks_base = graph.get_pageranks()

    result_from_coo, pageranks_coo = compute_time(compute_pagerank_from_coo, graph_coo=coo, damping_factor=args.damping_factor, max_iter=args.max_iter, return_results=True)

    n_cores = int(args.n_cores)
    result_parallel, pageranks_parallel = compute_time(
        compute_pagerank_joblib, 
        graph_coo=coo, 
        n_cores=n_cores, 
        damping_factor=args.damping_factor, 
        max_iter=args.max_iter, 
        return_results=True)

    rich.print(f"{result_base=}")
    rich.print(f"{result_from_coo=}")
    rich.print(f"{result_parallel=}")

    mse = {
        "coo": compute_mse(pageranks_base, pageranks_coo),
        "parallel": compute_mse(pageranks_base, pageranks_parallel)
    }

    rich.print(f"{mse=}")

    if args.save:

        if args.from_egdes: 
            figure_file=f"figs/graph_{args.n_edges}edges.png"
            save_path=f"backup/pageranks_{args.n_edges}edges.json"
        else:
            figure_file=f"figs/graph_{args.n_nodes}nodes_{args.min_conn_per_node}min_conn_per_node.png"
            save_path=f"backup/pageranks_{args.n_nodes}nodes_{args.min_conn_per_node}min_conn_per_node.json"

        pageranks = pageranks_to_dict(graph, save_path) 
        rich.print(f"Updated PR values with {damping_factor=} after {max_iter=} iterations: {pageranks}") 
        visualize_graph(graph, figure_file=figure_file)


parser = argparse.ArgumentParser()
parser.add_argument("-fe", "--from_egdes", default=False)
parser.add_argument("-ne", "--n_edges", default=10)
parser.add_argument("-nn", "--n_nodes", default=100)
parser.add_argument("-mc", "--min_conn_per_node", default=5)
parser.add_argument("-df", "--damping_factor", default=0.85)
parser.add_argument("-mi", "--max_iter", default=100)
parser.add_argument("-save", "--save", default=False)
parser.add_argument("-nc", "--n_cores", default=-1)

args = parser.parse_args()
main(args)