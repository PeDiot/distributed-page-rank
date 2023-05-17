import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import scipy.sparse

pagerank_kernel_code = """
__global__ void pagerank_kernel(float* pagerank_vector, const int *edges, const int *indptr, 
                                const int *out_degree, const float damping_factor, const int num_nodes) {   

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        float sum = 0.0f;
        for (int j = indptr[tid]; j < indptr[tid+1]; j++) {
            const int neighbor = edges[j];
            const int neighbor_degree = out_degree[neighbor];
            sum += pagerank_vector[neighbor] / neighbor_degree;
        }
        pagerank_vector[tid] = damping_factor * sum + (1.0f - damping_factor) / num_nodes;
    }
}
"""

def compute_pagerank_gpu(graph_coo: scipy.sparse.coo_matrix, damping_factor: float=0.85, max_iter: int=100, block_size: int=32):
    """Compute the pagerank of a graph using the GPU parallelization. 
    
    Args:
        graph_coo (scipy.sparse.coo_matrix): The graph in COO format.
        damping_factor (float, optional): The damping factor. Defaults to 0.85.
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        block_size (int, optional): The block size for the GPU kernel. Defaults to 256."""

    num_nodes = graph_coo.shape[0]
    graph = graph_coo.tocsr()

    edges = graph.indices.astype(np.int32)
    indptr = graph.indptr.astype(np.int32)
    out_degree = np.diff(indptr).astype(np.int32)

    # Put graph attributes on GPU
    edges_gpu, indptr_gpu, out_degree_gpu = cuda.In(edges), cuda.In(indptr), cuda.In(out_degree)

    pagerank_vector = np.ones(num_nodes, dtype=np.float32) / num_nodes
    # Create bidirectional data transfer between the CPU and GPU
    pagerank_vector_gpu = cuda.InOut(pagerank_vector)
    
    # Compile CUDA kernel and get block size
    pagerank_kernel = SourceModule(pagerank_kernel_code).get_function("pagerank_kernel")
    grid_size = (num_nodes + block_size - 1) // block_size

    for _ in range(max_iter):
        pagerank_kernel( 
            pagerank_vector_gpu, 
            edges_gpu,
            indptr_gpu, 
            out_degree_gpu,
            np.float32(damping_factor),
            np.int32(num_nodes), 
            block=(block_size, 1, 1), 
            grid=(grid_size, 1))
        
        cuda.Context.synchronize()

    pagerank_sum = np.sum(pagerank_vector)
    if pagerank_sum > 0:
        pagerank_vector /= np.sum(pagerank_vector)

    return pagerank_vector