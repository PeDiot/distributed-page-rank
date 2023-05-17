import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import scipy.sparse
import numpy as np


def make_pagerank_kernel(num_nodes: int) -> cuda.Function: 
    """Creates the CUDA kernel for the PageRank algorithm.
    
    Parameters:
        num_nodes (int): The number of nodes in the graph.
        
    Returns:
        pycuda.driver.Function: The CUDA kernel for the PageRank algorithm."""

    pagerank_kernel_code = """
        __global__ void pagerank_kernel(float *pageranks, const int *edges, const int *indptr, const int *out_degrees, const float damping_factor)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int num_nodes = %(num_nodes)s;
            const float one_minus_damping = 1.0f - damping_factor;

            if (i < num_nodes) {
                float sum = 0.0f;
                for (int j = indptr[i]; j < indptr[i+1]; j++) {
                    const int neighbor = edges[j];
                    const int neighbor_degree = out_degrees[neighbor];
                    sum += pageranks[neighbor] / neighbor_degree;
                }
                pageranks[i] = one_minus_damping / num_nodes + damping_factor * sum;
            }
        }
    """ % {"num_nodes": num_nodes}
    pagerank_kernel = SourceModule(pagerank_kernel_code).get_function("pagerank_kernel")
    return pagerank_kernel

def compute_pagerank_gpu(graph_coo: scipy.sparse.coo_matrix, damping_factor: float=.85, max_iter: int=100):
    """Computes the PageRank score of each node in the graph with GPU parallelization using PyCUDA.

    Parameters:
        graph_coo (scipy.sparse.csr_matrix): The adjacency matrix of the graph in COO format.
        damping_factor (float): The damping factor. Default is 0.85.
        max_iter (int): The maximum number of iterations. Default is 100.
        tol (float): The tolerance for convergence. Default is 1e-6.

    Returns:
        numpy.ndarray: The PageRank scores of each node in the graph."""

    graph = graph_coo.tocsr()
    num_nodes = graph.shape[0]

    edges = graph.indices.astype(np.int32)
    indptr = graph.indptr.astype(np.int32)
    out_degrees = np.diff(indptr).astype(np.int32)

    pageranks = np.ones(num_nodes, dtype=np.float32) / num_nodes
    last_pageranks = np.zeros(num_nodes, dtype=np.float32)
    pagerank_kernel = make_pagerank_kernel(num_nodes)

    for _ in range(max_iter):
        pagerank_kernel(cuda.InOut(pageranks), cuda.In(edges), cuda.In(indptr), cuda.In(out_degrees), np.float32(damping_factor))    
        last_pageranks[:] = pageranks
        
    pageranks_sum = pageranks.sum()
    if pageranks_sum > 0:
        pageranks /= pageranks_sum

    return pageranks