import scipy.sparse
import numpy as np
from joblib import Parallel, delayed
cimport numpy as np
cimport cython

CPU_COUNT = 8

@cython.boundscheck(False)
@cython.wraparound(False)
def pagerank_step(int node, int[:] indptr, int[:] indices, double[:] data, double[:] prev_x, double[:] out_degree, double damping_factor, int N):
    """Pagerank update rule for one node."""

    cdef:
        int j, col
        double sum = 0.0
    for j in range(indptr[node], indptr[node+1]):
        col = indices[j]
        sum += data[j] * prev_x[col] / out_degree[col]
        
    return damping_factor * sum + (1 - damping_factor) / N

def compute_pagerank_cython(graph_coo: scipy.sparse.coo_matrix, damping_factor: float=.85, num_threads: int=CPU_COUNT-1, max_iter: int=100, tol=1e-6):
    """Computes the PageRank score of each node in the graph using with parallelization using CPU threads.

    Args:
        graph_coo (scipy.sparse.coo_matrix): The adjacency matrix of the graph in COO format.
        damping_factor (float): The damping factor.
        max_iter (int): The maximum number of iterations. Default is 100.
        tol (float): The tolerance for convergence. Default is 1e-6.
        num_threads (int): The number of threads to use for parallelization. Default is the number of CPU cores.

    Returns:
        numpy.ndarray: The PageRank scores of each node in the graph."""
        
    graph_csr = graph_coo.tocsr()
    cdef int N = graph_csr.shape[0]

    out_degree = graph_csr.sum(axis=1).A.ravel().astype(np.float64)

    pr = np.full(N, 1/N, dtype=np.float64)

    if num_threads < 1:
        num_threads = None

    with Parallel(n_jobs=num_threads) as parallel:
        for _ in range(max_iter):
            prev_pr = pr.copy()
            results = parallel(delayed(pagerank_step)(node, graph_csr.indptr, graph_csr.indices, graph_csr.data, prev_pr, out_degree, damping_factor, N) for node in range(N))
            pr = np.array(results)

            if np.abs(prev_pr - pr).max() < tol:
                break

    pr_sum = pr.sum()
    if pr_sum > 0:
        pr /= pr_sum

    return pr