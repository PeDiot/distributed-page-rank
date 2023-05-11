from typing import List
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