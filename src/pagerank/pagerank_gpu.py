import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import scipy.sparse
import numpy as np


def make_pagerank_kernel(N: int) -> cuda.Function: 
    """Creates the CUDA kernel for the PageRank algorithm.
    
    Parameters:
        N (int): The number of nodes in the graph.
        
    Returns:
        pycuda.driver.Function: The CUDA kernel for the PageRank algorithm."""

    pagerank_kernel_code = """
        __global__ void pagerank_kernel(float *pagerank, const int *edges, const int *indptr, const int *out_degree, const float damping_factor)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int N = %(N)s;
            const float one_minus_damping = 1.0f - damping_factor;

            if (i < N) {
                float sum = 0.0f;
                for (int j = indptr[i]; j < indptr[i+1]; j++) {
                    const int neighbor = edges[j];
                    const int neighbor_degree = out_degree[neighbor];
                    sum += pagerank[neighbor] / neighbor_degree;
                }
                pagerank[i] = one_minus_damping / N + damping_factor * sum;
            }
        }
    """ % {"N": N}
    pagerank_kernel = SourceModule(pagerank_kernel_code).get_function("pagerank_kernel")

def compute_pagerank_gpu(graph: scipy.sparse.coo_matrix, damping_factor: float=.85, max_iter: int=100, tol: float=1e-6):
    """Computes the PageRank score of each node in the graph with GPU parallelization using PyCUDA.

    Parameters:
        graph (scipy.sparse.csr_matrix): The adjacency matrix of the graph in CSR format.
        damping_factor (float): The damping factor. Default is 0.85.
        max_iter (int): The maximum number of iterations. Default is 100.
        tol (float): The tolerance for convergence. Default is 1e-6.

    Returns:
        numpy.ndarray: The PageRank scores of each node in the graph."""

    # Convert the graph to the CSR format
    graph = graph.tocsr()
    N = graph.shape[0]

    edges = graph.indices.astype(np.int32)
    indptr = graph.indptr.astype(np.int32)
    out_degree = np.diff(indptr).astype(np.int32)

    pr = np.ones(N, dtype=np.float32) / N
    last_pr = np.zeros(N, dtype=np.float32)
    pagerank_kernel = make_pagerank_kernel(N)

    for _ in range(max_iter):
        pagerank_kernel(cuda.InOut(pr), cuda.In(edges), cuda.In(indptr), cuda.In(out_degree), np.float32(damping_factor)) 
        
        if np.abs(pr - last_pr, np.inf).max() < tol:
            break
    
        last_pr[:] = pr
        
    pr_sum = pr.sum()
    if pr_sum > 0:
        pr /= pr_sum

    return pr