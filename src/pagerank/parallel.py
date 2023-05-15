from joblib import Parallel, delayed
import scipy.sparse
import numpy as np

from .coo import pagerank_one_node


def compute_pagerank_joblib(graph_coo: scipy.sparse.coo.coo_matrix, n_cores: int=-1, damping_factor: float=.85, max_iter: int=100, tol: float=1e-6) -> np.ndarray:
    """Paralellize the PageRank score computation of each node in the graph from adjacency matrix in COO format using joblib.

    Args:
        graph_coo (scipy.sparse.coo_matrix): The adjacency matrix of the graph in COO format.
        n_cores (int): The number of cores to use. Default is -1, which means all the cores are used.
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
        pr = Parallel(n_jobs=n_cores)(delayed(pagerank_one_node)(node, graph_csr, prev_pr, out_degree, damping_factor, N) for node in range(N))
        pr = np.array(pr)

        if np.abs(pr - prev_pr).max() < tol:
            break

    return pr