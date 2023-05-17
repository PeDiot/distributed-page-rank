import scipy.sparse
import numpy as np  


def compute_pagerank_base(graph_coo: scipy.sparse.coo_matrix, max_iter: int, damping_factor: float) -> np.ndarray:
    """Computes the PageRank score of each node in the graph using NumPy.

    Args:
        graph_coo (scipy.sparse.coo_matrix): The adjacency matrix of the graph in COO format.
        max_iter (int): The maximum number of iterations. Default is 100.

    Returns:
        np.ndarray: The PageRank scores of each node in the graph."""

    graph_csr = graph_coo.tocsr()
    num_nodes = graph_csr.shape[0]

    out_degree = graph_csr.sum(axis=1).A.ravel().astype(np.float32)
    pageranks = np.full(num_nodes, 1/num_nodes, dtype=np.float32)

    for _ in range(max_iter):

        prev_pageranks = pageranks.copy()

        for node in range(num_nodes):
            sum = 0.0
            for neighbor in range(graph_csr.indptr[node], graph_csr.indptr[node+1]):
                col = graph_csr.indices[neighbor]
                sum += graph_csr.data[neighbor] * prev_pageranks[col] / out_degree[col]

            pageranks[node] = damping_factor * sum + (1 - damping_factor) / num_nodes

    return pageranks