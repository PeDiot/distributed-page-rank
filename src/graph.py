from typing import Union, List, Optional, Dict
import numpy as np
import scipy.sparse

from src.node import Node


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

def generate_random_adjacency_matrix(n_nodes: int, min_conn_per_node: int) -> scipy.sparse.coo_matrix: 
    """Generates a connected graph with a specified number of nodes and minimum number of connections per node using a random algorithm.

    Parameters:
        n_nodes (int): The number of nodes in the graph.
        min_conn_per_node (int): The minimum number of connections per node.

    Returns:
        scipy.sparse.coo_matrix: The adjacency matrix of the generated graph in COO format."""

    graph = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        graph[i, i] = 1

    while True:
        if (graph.sum(axis=1) >= min_conn_per_node).all():
            break

        node = np.random.choice(np.where(graph.sum(axis=1) < min_conn_per_node)[0])

        while graph[node].sum() < min_conn_per_node:
            connect_to = np.random.randint(0, n_nodes)

            if graph[node, connect_to] == 1 or graph[connect_to, node] == 1:
                continue

            graph[node, connect_to] = 1
            graph[connect_to, node] = 1

    coo = scipy.sparse.coo_matrix(graph)
    return coo

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
    
    @classmethod
    def from_adjacency_matrix(cls, adjacency_matrix: np.ndarray) -> "Graph":
        """Returns a graph object from an adjacency matrix."""
        graph = cls()

        for i in range(adjacency_matrix.shape[0]):
            parent = i
            for j in range(i + 1, adjacency_matrix.shape[1]):
                child = j
                if adjacency_matrix[i, j] == 1:
                    graph.add_edge(parent, child)

        graph.sort_nodes()
        return graph

    def sort_nodes(self):
        self.nodes.sort(key=lambda node: int(node.name))

    def get_pageranks(self) -> np.ndarray:
        return np.array([node.pagerank for node in self.nodes])

    def normalize_pagerank(self):
        pagerank_sum = np.sum(self.get_pageranks())

        for node in self.nodes:
            node.pagerank /= pagerank_sum