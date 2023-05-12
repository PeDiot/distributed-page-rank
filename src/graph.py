from typing import Union, List, Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

    def visualize(self, figure_file: Optional[str] = None):
        """Visualize the graph using networkx and matplotlib."""

        G = nx.DiGraph()
        node_labels = {}

        for node in self.nodes:
            G.add_node(node.name)
            node_labels[node.name] = f"{node.name}\nPR={node.pagerank:.2f}"
            for child in node.children:
                G.add_edge(node.name, child.name)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)

        if figure_file:
            plt.savefig(figure_file)
            
        plt.show()

    def sort_nodes(self):
        self.nodes.sort(key=lambda node: int(node.name))

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def get_pagerank_list(self):
        pagerank_list = np.asarray([node.pagerank for node in self.nodes], dtype='float32')
        return np.round(pagerank_list, 3)
