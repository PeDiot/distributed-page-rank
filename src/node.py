from typing import Union
import numpy as np


class Node:
    """Node object with name, children, parents, and pagerank attributes."""
    def __init__(self, name: Union[int, str]):
        self.name = name
        self.children = []
        self.parents = []
        self.pagerank = 1.0

    def __repr__(self) -> str:
        return f"Node(name={self.name}, children={self.children}, parents={self.parents}, pagerank={self.pagerank})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def link_child(self, new_child: "Node"):
        for child in self.children:
            if(child.name == new_child.name):
                return None
        self.children.append(new_child)

    def link_parent(self, new_parent: "Node"):
        for parent in self.parents:
            if(parent.name == new_parent.name):
                return None
        self.parents.append(new_parent)

    def update_pagerank(self, d: float, n: int):
        """Update the pagerank of the node.
        
        Args:
            d (float): Damping factor, probability of randomly walking out the links.
            n (int): Number of nodes in the graph."""
        
        in_neighbors = self.parents
        pagerank_sum = sum((node.pagerank / len(node.children)) for node in in_neighbors)
        random_jumping = d / n
        self.pagerank = random_jumping + (1-d) * pagerank_sum