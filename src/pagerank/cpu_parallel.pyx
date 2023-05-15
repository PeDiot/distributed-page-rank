import numpy as np
from numpy import float32_t

cimport numpy as np
cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compute_pagerank_with_cython(np.ndarray[long, ndim=1] indptr, np.ndarray[long, ndim=1] indices, np.ndarray[np.float32_t, ndim=1] data, int N, float damping_factor, int max_iter, float tol):
    cdef int i, j, iter
    cdef np.ndarray[np.float32_t, ndim=1] pr = np.full(N, 1/N, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] prev_pr = np.empty_like(pr)

    cdef np.ndarray[np.float32_t, ndim=1] out_degree = np.zeros(N, dtype=np.float32)
    cdef np.float32_t sum_pr = 0.0

    # Compute out_degree
    for i in range(N):
        for j in range(indptr[i], indptr[i+1]):
            out_degree[i] += data[j]

    # PageRank iteration
    for iter in range(max_iter):
        sum_pr = 0.0
        prev_pr[:] = pr

        with nogil, prange(N) as prange_idx, parallel(num_threads=4):
            for i in prange_idx:
                local_sum_pr = 0.0

                for j in range(indptr[i], indptr[i+1]):
                    local_sum_pr += data[j] * pr[indices[j]] / out_degree[indices[j]]

                pr[i] = damping_factor * local_sum_pr + (1 - damping_factor) / N
                sum_pr += pr[i]

        with nogil:
            if np.abs(prev_pr - pr).max() < tol:
                break

    return pr
