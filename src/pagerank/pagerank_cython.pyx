import scipy.sparse
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double pagerank_step(
    int node, 
    int[:] indptr, 
    int[:] indices, 
    double[:] data, 
    double[::1] pageranks, 
    double[:] out_degrees, 
    double damping_factor, 
    int num_nodes
) nogil:

    cdef:
        int neighbor, col
        double sum = 0.0

    for neighbor in range(indptr[node], indptr[node+1]):
        col = indices[neighbor]
        sum += data[neighbor] * pageranks[col] / out_degrees[col]

    return damping_factor * sum + (1 - damping_factor) / num_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_pageranks(
    int[:] nodes, 
    int[:] indptr, 
    int[:] indices, 
    double[:] data, 
    double[::1] pageranks, 
    double[:] out_degrees, 
    double damping_factor, 
    int max_iter, 
    int num_threads
):
    cdef int i, node, iter
    cdef int num_nodes = len(nodes)
    cdef double[::1] temp_pageranks = np.zeros(num_nodes, dtype=np.float64)

    for iter in range(max_iter):

        for i in prange(num_nodes, nogil=True, num_threads=num_threads):
            node = nodes[i]
            temp_pageranks[i] = pagerank_step(node, indptr, indices, data, pageranks, out_degrees, damping_factor, num_nodes)

        for i in range(len(nodes)):
            pageranks[i] = temp_pageranks[i]


def compute_pagerank_cython(graph_coo: scipy.sparse.coo_matrix, num_threads: int, damping_factor: float=.85, max_iter: int=100):
    """Computes the PageRank score of each node in the graph using with parallelization using CPU threads.

    Args:
        graph_coo (scipy.sparse.coo_matrix): The adjacency matrix of the graph in COO format.
        damping_factor (float): The damping factor.
        max_iter (int): The maximum number of iterations. Default is 100.
        num_threads (int): The number of threads to use for parallelization. Default is the number of CPU cores.

    Returns:
        The PageRank scores of each node in the graph."""
        
    graph_csr = graph_coo.tocsr()
    cdef int num_nodes = graph_csr.shape[0]
    cdef nodes = np.arange(num_nodes, dtype=np.int32)

    out_degree = graph_csr.sum(axis=1).A.ravel().astype(np.float64)
    cdef double[::1] pageranks = np.zeros(num_nodes, dtype=np.float64)

    compute_pageranks(
        nodes,
        graph_csr.indptr,
        graph_csr.indices,
        graph_csr.data,
        pageranks,
        out_degree,
        damping_factor,
        max_iter, 
        num_threads)

    pagerank_sum = np.sum(pageranks)
    if pagerank_sum > 0:
        pageranks /= pagerank_sum

    return pageranks