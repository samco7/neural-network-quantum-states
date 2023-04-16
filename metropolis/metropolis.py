import numpy as np
from tqdm import tqdm

class MetropolisIsing:
    def __init__(self, L, J=1, mu=1, h=0, periodic=True):
        """
            Constructor. Builds the lattice and gets the initial energy. Saves attributes. Assumes constant field h.
        """
        self.J = J
        self.mu = mu
        self.h = h
        self.lattice = np.random.choice([-1, 1], size=(L, L))
        self.L = L
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                if j + 1 < self.L: energy -= self.lattice[i, j]*self.lattice[i, j + 1]
                if i + 1 < self.L: energy -= self.lattice[i, j]* self.lattice[i + 1, j]
        if periodic:
            for i in range(self.L):
                energy -= self.lattice[-1, i]*self.lattice[0, i]
                energy -= self.lattice[i, -1]*self.lattice[i, 0]

        self.energy = energy*self.J

        for entry in self.lattice.ravel():
            self.energy -= self.mu*self.h*entry

    def compute_magnetization(self):
        """
            Computes the magnetization as the mean spin of the lattice.
        """
        return np.mean(self.lattice)

    def compute_energy_difference(self, flip_indices):
        """
            Get the energy difference resulting from flipping the spin at flip_indices.
        """
        # unpack the indices
        i, j = flip_indices

        # get the indices of all neighbors
        neighbor_indices = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]
        if i == self.L - 1:
            neighbor_indices.remove((i + 1, j))
            neighbor_indices.append((0, j))
        if j == self.L - 1:
            neighbor_indices.remove((i, j + 1))
            neighbor_indices.append((i, 0))

        # compute and return the energy difference for the interactions involving this particle
        original_local_contribution = -self.J*sum(
            [self.lattice[i, j]*self.lattice[indices] for indices in neighbor_indices]) - self.mu*self.h*self.lattice[i, j]
        new_local_contribution = -self.J*sum(
            [-1*self.lattice[i, j]*self.lattice[indices] for indices in neighbor_indices]) + self.mu*self.h*self.lattice[i, j]
        return new_local_contribution - original_local_contribution

    def metropolis_step(self, beta):
        """
            Carries out a single step of Metropolis for a given value of beta.
        """
        # propose to flip the spin at a random set of indices
        flip_indices = tuple(np.random.randint(0, self.L, 2))

        # get the energy difference between the current and proposed configurations
        energy_difference = self.compute_energy_difference(flip_indices)

        # create a mask to use in the cases where the proposed configuration is accepted
        mask = np.ones((self.L, self.L))
        mask[flip_indices] = -1

        # if the energy difference is negative, accept the proposed configuration
        if energy_difference < 0:
            self.lattice = self.lattice*mask
            self.energy += energy_difference

        # if the energy difference is nonnegative, accept the proposed configuration with probability exp(−β∆E)
        elif np.random.uniform() < np.exp(-beta*energy_difference):
            self.lattice = self.lattice*mask
            self.energy += energy_difference

    def metropolis(self, T, n_steps=100000, burn_in=20000):
        """
            For a temperature  T, carries out n_steps step of Metropolis, saving samples after burn_in iterations.
        """
        # initialize lists to hold values, and set beta to be 1/T (scale so k_B is 1)
        magnetizations = []
        energies = []
        beta = 1/T

        # iterate through n_steps
        for i in range(n_steps):
            self.metropolis_step(beta)

            # if thermalized, start saving magnetization and energy samples
            if i > burn_in:
                magnetizations.append(np.abs(self.compute_magnetization()))
                energies.append(self.energy)

        # get the expected values for key quantities, and return them
        expected_abs_mag = np.mean(np.abs(magnetizations))
        expected_mag_sus = beta*(np.mean(np.array(magnetizations)**2/self.L**2) - expected_abs_mag**2/self.L**2)
        expected_energy = np.mean(energies)
        expected_spec_heat = np.var(energies)*beta**2
        return expected_abs_mag, expected_mag_sus, expected_energy, expected_spec_heat
