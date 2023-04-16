import numpy as np

class ClassicIsing(object):
    """
    """

    def __init__(self, L, J=1, mu=1, h=0, min_flip=1, periodic=True):
        self.L = L
        self.n_spins = self.L**2
        self.periodic = periodic
        self.J = J
        self.mu = mu
        self.h = h
        self.min_flip = min_flip

    def min_flips(self):
        return self.min_flip

    def num_spins(self):
        return self.n_spins

    def field(self):
        return self.h

    def is_periodic(self):
        return self.periodic

    def find_matrix_elements(self, state):
        lattice = np.array(state).reshape((self.L, self.L))
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                if j + 1 < self.L:
                    energy -= lattice[i, j]*lattice[i, j + 1]
                if i + 1 < self.L:
                    energy -= lattice[i, j]* lattice[i + 1, j]
        if self.periodic:
            for i in range(self.L):
                energy -= lattice[-1, i]*lattice[0, i]
                energy -= lattice[i, -1]*lattice[i, 0]
        energy *= self.J

        for entry in state:
            energy -= self.mu*self.h*entry

        matrix_elements = [energy]
        spin_flip_transitions = [[]]

        return matrix_elements, spin_flip_transitions
