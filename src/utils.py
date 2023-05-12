from typing import List
import time 
import numpy as np
from tqdm import tqdm


def file_to_edges(fname) -> List:
    """Returns a list of edges from txt file."""

    if not fname.endswith(".txt"):
        raise ValueError("File must be a .txt file")

    with open(fname) as f:
        lines = f.readlines()

    edges = [
        line.strip().split(",") for line in lines
    ]
    return edges


def compute_time(func, n_repeat: int=50, return_results: bool=False, *args, **kwargs):
    """Returns the time taken to run a function."""

    times = []
    loop = tqdm(range(n_repeat))

    for _ in loop:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
    
    res = {
        "min": min(times),
        "max": max(times),
        "mean": np.mean(times),
        "median": np.median(times), 
        "std": np.std(times)
    }

    if return_results:
        return res, result
    return res

def generate_random_edges(n_edges: int) -> List: 
    """Returns a list of random edges."""

    edges = []
    i = 0
    while i < n_edges:
        x = np.random.randint(0, n_edges)
        y = np.random.randint(0, n_edges)

        while x == y:
            y = np.random.randint(0, n_edges)
 
        if [x, y] not in edges:
            edges.append([x, y])
            i += 1

    return edges
