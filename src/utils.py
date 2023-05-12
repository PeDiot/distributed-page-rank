from typing import List
import time 
from src.graph import Graph


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

def compute_time(func, return_results: bool=False, *args, **kwargs):
    """Returns the time taken to run a function."""

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if return_results:
        return elapsed_time, result
    return elapsed_time
