import scipy.sparse
import numpy as np  


def pagerank_update(neighbors: np.ndarray, pr: np.ndarray, out_degree: np.ndarray, damping_factor: float, N: int) -> float:
    """Computes the PageRank score of a node using numpy.

    Args:
        neighbors (np.ndarray): The neighbors of the node.
        pr (np.ndarray): The PageRank scores of the neighbors of the node.
        out_degree (np.ndarray): The out degree of the node.
        damping_factor (float): The damping factor.
        N (int): The number of nodes in the graph.

    Returns:
        float: The PageRank score of the node."""
    
    out_pr = np.multiply(neighbors, pr) / out_degree
    return damping_factor * np.sum(out_pr) + (1 - damping_factor) / N

def pagerank_one_node(node: int, graph_csr: scipy.sparse.csr.csr_matrix, pr: np.ndarray, out_degree: np.ndarray, damping_factor: float, N: int): 
    """Updates the PageRank score of a node.

    Args:
        node (int): The node index.
        graph_csr (scipy.sparse.csr.csr_matrix): The adjacency matrix of the graph in CSR format.
        pr (np.ndarray): The PageRank scores of the nodes.
        out_degree (np.ndarray): The out degree of the nodes.
        damping_factor (float): The damping factor.
        N (int): The number of nodes in the graph.
    
    Returns:
        float: The updated PageRank score of the node."""

    row_indices = slice(graph_csr.indptr[node], graph_csr.indptr[node+1])
    neighbors = graph_csr.data[row_indices]
    pr_neighbors = pr[graph_csr.indices[row_indices]]
    out_degree_neighbors = out_degree[graph_csr.indices[row_indices]]

    pr_node = pagerank_update(neighbors, pr_neighbors, out_degree_neighbors, damping_factor, N)
    return pr_node 

def compute_pagerank_numpy(graph_coo: scipy.sparse.coo.coo_matrix, damping_factor: float=.85, max_iter: int=100) -> np.ndarray:
    """Computes the PageRank score of each node in the graph from adjacency matrix in COO format.
    
    Args:
        graph_coo (scipy.sparse.coo_matrix): The adjacency matrix of the graph in COO format.
        damping_factor (float): The damping factor. Default is 0.85.
        max_iter (int): The maximum number of iterations. Default is 100.
        tol (float): The tolerance for convergence. Default is 1e-6.

    Returns:
        np.ndarray: The PageRank scores of each node in the graph."""

    graph_csr = graph_coo.tocsr()
    N = graph_csr.shape[0]

    out_degree = graph_csr.sum(axis=1).A.ravel().astype(np.float32)
    pr = np.full(N, 1/N, dtype=np.float32)

    for _ in range(max_iter):
        prev_pr = pr.copy()

        for node in range(N):
            pr[node] = pagerank_one_node(node, graph_csr, pr, out_degree, damping_factor, N)

    return pr
