import numpy as np
import scipy.sparse
import random 


def generate_random_adjacency_matrix(n_nodes: int, min_conn_per_node: float, undirected_prob: float=.5) -> scipy.sparse.coo_matrix: 
    """Generates a connected graph with a specified number of nodes and minimum number of connections per node using a random algorithm.

    Parameters:
        n_nodes (int): The number of nodes in the graph.
        min_conn_per_node (float): The proportion of nodes each node is connected to.
        undirected_prob (float): The probability of an edge being undirected.       

    Returns:
        scipy.sparse.coo_matrix: The adjacency matrix of the generated graph in COO format."""

    graph = np.zeros((n_nodes, n_nodes))

    while True:
        prop = graph.sum(axis=1) / n_nodes

        if (prop >= min_conn_per_node).all():
            break

        node = np.random.choice(np.where(prop < min_conn_per_node)[0])

        while graph[node].sum() / n_nodes < min_conn_per_node:
            connect_to = np.random.randint(0, n_nodes)

            if graph[node, connect_to] == 1 or graph[connect_to, node] == 1:
                continue

            graph[node, connect_to] = 1

            if random.random() > undirected_prob: 
                graph[connect_to, node] = 1

    coo = scipy.sparse.coo_matrix(graph)
    return coo