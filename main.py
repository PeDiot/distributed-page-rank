from src.utils import (
    compute_time, 
    pageranks_to_dict, 
    visualize_graph
) 
from src.graph import Graph, generate_random_adjacency_matrix, generate_random_edges
from src.pagerank.base import compute_pagerank
from src.pagerank.coo import compute_pagerank_from_coo
# from src.pagerank.cpu_parallel import compute_pagerank_parallel

import rich
import argparse

def main(args): 
    if args.from_egdes: 
        edges = generate_random_edges(args.from_edges)
        graph = Graph.from_edges(edges)
        rich.print(f"{args.n_edges} random edges in the graph.")
    else:
        coo = generate_random_adjacency_matrix(args.n_nodes, args.min_conn_per_node)
        graph = Graph.from_adjacency_matrix(coo.toarray())
        rich.print(f"{args.n_nodes} random nodes in the graph with a minimum of {args.min_conn_per_node} connections per node.")

    result_base = compute_time(compute_pagerank, graph=graph, damping_factor=args.damping_factor, max_iter=args.max_iter) 
    result_from_coo = compute_time(compute_pagerank_from_coo, graph_coo=coo, damping_factor=args.damping_factor, max_iter=args.max_iter)
    # result_cpu_parallel = compute_time(
    #     compute_pagerank_with_cython, 
    #     indptr=coo.indptr,
    #     indices=coo.indices,
    #     data=coo.data,
    #     damping_factor=args.damping_factor, 
    #     max_iter=args.max_iter)

    rich.print(f"{result_base=}")
    rich.print(f"{result_from_coo=}")
    # rich.print(f"{result_cpu_parallel=}")

    if args.save:

        if args.from_egdes: 
            figure_file=f"figs/graph_{args.n_edges}edges.png"
            save_path=f"backup/pageranks_{args.n_edges}edges.json"
        else:
            figure_file=f"figs/graph_{args.n_nodes}nodes_{args.min_conn_per_node}min_conn_per_node.png"
            save_path=f"backup/pageranks_{args.n_nodes}nodes_{args.min_conn_per_node}min_conn_per_node.json"

        pageranks = pageranks_to_dict(graph, save_path) 
        rich.print(f"Updated PR values with {args.damping_factor=} after {args.n_iter=} iterations: {pageranks}") 
        visualize_graph(graph, figure_file=figure_file)


parser = argparse.ArgumentParser()
parser.add_argument("-fe", "--from_egdes", default=False)
parser.add_argument("-ne", "--n_edges", default=10)
parser.add_argument("-nn", "--n_nodes", default=100)
parser.add_argument("-mc", "--min_conn_per_node", default=5)
parser.add_argument("-df", "--damping_factor", default=0.85)
parser.add_argument("-mi", "--max_iter", default=100)
parser.add_argument("-save", "--save", default=False)

args = parser.parse_args()
main(args)