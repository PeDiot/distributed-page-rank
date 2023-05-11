from typing import Union, List
import numpy as np

from src.node import Node


class Graph:
    def __init__(self):
        self.nodes = []

    def __repr__(self) -> str:
        return f"Graph(nodes={self.nodes})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def contains(self, name: Union[int, str]):
        for node in self.nodes:
            if(node.name == name):
                return True
        return False

    def find(self, name: Union[int, str]):
        """Return the node with the name, create and return new node if not found."""
        if not self.contains(name):
            new_node = Node(name)
            self.nodes.append(new_node)
            return new_node
        else:
            return next(node for node in self.nodes if node.name == name)

    def add_edge(self, parent: Node, child: Node):
        parent_node = self.find(parent)
        child_node = self.find(child)

        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)
        
    @classmethod
    def from_edges(cls, edges: List) -> "Graph":
        """Returns a graph object from a list of edges."""
        graph = cls()

        for edge in edges:
            parent, child = edge
            graph.add_edge(parent, child)

        graph.sort_nodes()
        return graph

    def display(self):
        for node in self.nodes:
            print(f"{node.name} links to {[child.name for child in node.children]}")

    def sort_nodes(self):
        self.nodes.sort(key=lambda node: int(node.name))

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def get_pagerank_list(self):
        pagerank_list = np.asarray([node.pagerank for node in self.nodes], dtype='float32')
        return np.round(pagerank_list, 3)
