import networkx as nx
import numpy as np

class Max_Cut_Ising(object):
    """
    """

    def __init__(self, graph, min_flip=1):
        """transverse field h_x"""
        self.n_spins = len(graph.nodes())
        self.graph = graph
        self.h_x = 0
        self.min_flip = min_flip
        self.matrix_elements = np.zeros((self.n_spins + 1)) - self.h_x
        self.spin_flip_transitions = [[]] + [[i] for i in range(self.n_spins)]

    def min_flips(self):
        return self.min_flip

    def num_spins(self):
        return self.n_spins

    def field(self):
        return self.h_x

    # def is_periodic(self):
    #     return self.periodic

    def find_matrix_elements(self, state):
        matrix_elements = self.matrix_elements
        matrix_elements[0] = 0.0

        for edge in self.graph.edges():
            weight = nx.get_edge_attributes(self.graph, 'weight')
            matrix_elements[0] += weight[edge]*state[edge[0]]*state[edge[1]]

        return matrix_elements, self.spin_flip_transitions
