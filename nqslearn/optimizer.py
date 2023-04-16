# Python implementation to optimize Neural Quantum States (NQS) from the paper
# _____________________________________________________________________________
# | G. Carleo, and M. Troyer                                                  |
# | Solving the quantum many-body problem with artificial neural-networks     |
# |___________________________________________________________________________|

import numpy as np
import pickle
import time
import os
from .sampler import MetropolisSampler


class SRoptimizer(object):
    """
    Optimizes the parameters of a neural quantum state (NQS) with respect to
    a Hamiltonian.
    """
    def __init__(self, nqs, ham, learning_rate=1e-2, decay=1e-4,
                 sampler_params=None, save=False):
        if not sampler_params:
            sampler_params = {'n_sweeps': 10000, 'therm_factor': 0.,
                              'sweep_factor': 1, 'n_flips': None}
        self.nqs = nqs
        self.ham = ham
        self.n_sweeps = sampler_params['n_sweeps']
        self.therm_factor = sampler_params['therm_factor']
        self.sweep_factor = sampler_params['sweep_factor']
        self.n_flips = sampler_params['n_flips']
        self.learning_rate = learning_rate
        self.decay = decay
        self.save = save
        self.loss = []
        self.error = []
        self.end_states = None
        if self.save:
            data_dir = './data_' + str(time.time()).split('.')[0] + '/'
            self.data_dir = data_dir
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            with open(self.data_dir+'sampler_params.pkl', 'wb') as f:
                pickle.dump(sampler_params, f)
            with open(self.data_dir+'hamiltonian.pkl', 'wb') as f:
                pickle.dump(ham, f)
            np.savetxt(self.data_dir+'learning_rate.txt',
                    np.array([learning_rate]))
        print('Optimizing the Hamiltonian:', ham)
        print('Sampler parameters:', sampler_params)
        print('Learning rate:', learning_rate)

    def run(self, num_epochs, sluggish=None):
        initial_state = None
        best_loss = np.inf
        counter = 0
        for p in range(num_epochs):
            print('\n\nIteration number:', p)
            sampler = MetropolisSampler(self.ham, self.nqs,
                                        zero_magnetization=False,
                                        initial_state=initial_state)
            sampler.run(self.n_sweeps, self.therm_factor, self.sweep_factor,
                        self.n_flips)
            initial_state = sampler.current_state
            update_vals, _ = self.compute_sr_gradients(sampler, p+1)
            rate = self.learning_rate*(1 / (1 + self.decay*p))
            if p < 5:
                rate /= 100
            self.nqs.update_params(rate*update_vals)

            # keep track of some stuff
            if self.save:
                np.savetxt(self.data_dir+'loss.txt',
                        np.squeeze(np.array(self.loss)))
                np.savetxt(self.data_dir+'error.txt',
                        np.squeeze(np.array(self.error)))
                with open(self.data_dir+'NQS_'+str(p)+'.pkl', 'wb') as f:
                    pickle.dump(self.nqs, f)
                with open(self.data_dir+'sampler_'+str(p)+'.pkl', 'wb') as f:
                    pickle.dump(sampler, f)
            if self.loss[-1] > best_loss:
                counter += 1
            else:
                counter = 0
                best_loss = self.loss[-1]
            if sluggish is not None and counter >= sluggish:
                break
            self.end_states = sampler.state_history

    # def cd(self, sampler, p):
    #     self.loss.append(sampler.nqs_energy)
    #     self.error.append(sampler.nqs_energy_err)
    #     sigmoid = lambda x: 1/(1 + np.exp(-x))
    #     states = sampler.state_history
    #     a = self.nqs.a
    #     b = self.nqs.b
    #     W = self.nqs.W
    #     dW = np.zeros((self.nqs.n_visible, self.nqs.n_hidden))
    #     da = np.zeros(len(a))
    #     db = np.zeros(len(b))

    #     for v in states:
    #         h = -np.ones(self.nqs.n_hidden)
    #         for i in range(len(h)):
    #             prob = sigmoid(b[i] + sum([v[j]*W[j, i] for j in range(self.nqs.n_visible)]))
    #             if np.random.uniform(0, 1) < prob: h[i] = 1

    #         v_new = -np.ones(self.nqs.n_visible)
    #         for j in range(len(v_new)):
    #             prob = sigmoid(a[j] + sum([h[i]*W[j, i] for i in range(self.nqs.n_hidden)]))
    #             if np.random.uniform(0, 1) < prob: v_new[i] = 1

    #         h_new = -np.ones(self.nqs.n_visible)
    #         for i in range(len(h_new)):
    #             prob = sigmoid(b[i] + sum([v_new[j]*W[j, i] for j in range(self.nqs.n_visible)]))
    #             if np.random.uniform(0, 1) < prob: h_new[i] = 1

    #         dW += (np.outer(v, h) - np.outer(v_new, h_new))
    #         da += v - v_new
    #         db += h - h_new

    #     return da.reshape(self.nqs.n_visible, 1), db.reshape(self.nqs.n_hidden, 1), dW

    def compute_sr_gradients(self, sampler, p):
        """
        Computes gradients based on the parameters derivatives and stochastic
        reconfiguration method
        """
        states = sampler.state_history
        energies = [np.squeeze(entry) for entry in sampler.local_energies]
        Elocs = np.array(energies).reshape(
            (len(sampler.local_energies), 1)
        )

        print('Computing stochastic reconfiguration updates')
        self.loss.append(sampler.nqs_energy)
        self.error.append(sampler.nqs_energy_err)
        B = self.nqs.b
        weights = self.nqs.W
        update, derivs = self.compute_derivs(
            p, Elocs, B, weights, np.array(states, dtype=complex),
            self.nqs.n_visible, self.nqs.n_hidden, len(states)
        )
        print('Completed stochastic reconfiguration updates')
        return update, derivs

    def compute_derivs(self, p, Eloc, B, weights, sigmas, n_spins, n_h, N_s):
        """Computes variational derivatives and update to params"""
        theta = np.dot(weights.transpose(),
                       sigmas.transpose()) + B  # n_h x N_s
        dA = sigmas.transpose()  # n_spins x N_s
        dB = np.tanh(theta)  # n_h x N_s
        dW = sigmas.transpose().reshape((n_spins, 1, N_s)) * \
             np.tanh(theta.reshape((1, n_h, N_s)))
        derivs = np.concatenate([dA, dB, dW.reshape(n_spins * n_h, N_s)])

        avg_derivs = np.sum(derivs, axis=1, keepdims=True) / N_s
        avg_derivs_mat = np.conjugate(avg_derivs.reshape(derivs.shape[0], 1))
        avg_derivs_mat = avg_derivs_mat * avg_derivs.reshape(
            1, derivs.shape[0]
        )
        moment2 = np.einsum('ik,jk->ij', np.conjugate(derivs), derivs) / N_s
        S_kk = np.subtract(moment2, avg_derivs_mat)

        F_p = np.sum(Eloc.transpose() * np.conjugate(derivs), axis=1) / N_s
        F_p -= np.sum(Eloc.transpose(), axis=1) * \
               np.sum(np.conjugate(derivs), axis=1) / (N_s ** 2)
        S_kk2 = np.zeros(S_kk.shape, dtype=complex)
        row, col = np.diag_indices(S_kk.shape[0])
        S_kk2[row, col] = self.lambd(p) * np.diagonal(S_kk)
        S_reg = S_kk + S_kk2
        if np.linalg.matrix_rank(S_reg) == S_reg.shape[0]:
            update = self.gamma(p)*np.dot(np.linalg.inv(S_reg), F_p).reshape(derivs.shape[0], 1)
        else:
            update = self.gamma(p)*np.dot(np.linalg.pinv(S_kk), F_p).reshape(derivs.shape[0], 1)

        return update, derivs

    @staticmethod
    def lambd(p, lambd0=100, b=0.9, lambdMin=1e-4):
        """
        Lambda regularization parameter for S_kk matrix,
        see supplementary materials
        """
        return max(lambd0 * (b ** p), lambdMin)

    @staticmethod
    def gamma(p, b=.999, gammaMin=0):
        return max(b**p, gammaMin)
