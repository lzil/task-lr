import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl
import os
import itertools
import sys
import argparse

import errno

from oneibl.one import ONE
from ibllib.misc import pprint

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

import pdb

sys.path.insert(1, '../')

from tools import *

eps = 1e-6

from one_tools import *
from one_compute import *

cache = 'cache'
figures = 'figures'

debug = False



dummy_data = []

#states = 3


# cues: contrast, prev choices, feedback



def calculate_nll(dt):
    state = 0
    ll = 0


class glmhmm:
    def __init__(self, dts):
        self.dts = [dts[1] for i in dts]



class hmm:
    def __init__(self, states=2):
        self.n_states = states

        # normalized transition matrix
        # starting state x ending state
        m_trans = np.random.rand(self.n_states, self.n_states)
        self.m_trans = (m_trans / np.sum(m_trans, axis=0)).T

        # output matrix
        m_out = np.random.rand(self.n_states, 2)
        self.m_out = (m_out.T / np.sum(m_out, axis=1)).T

        m_init = np.random.rand(self.n_states)
        self.m_init = m_init / np.sum(m_init)
    


    def _EM_step(self, dts):
        n_sessions = len(dts)
        
        _gammas = np.zeros((self.n_states))
        _gammas_minus_one = np.zeros((self.n_states))
        _xi = np.zeros((self.n_states,self.n_states))
        _inits = np.zeros((self.n_states))
        _gamma_one_counts = np.zeros((self.n_states))

        px = np.zeros((n_sessions))

        for idx, sess in enumerate(dts):

            choice = sess.choice
            choice[np.where(choice == 1)[0]] = 0
            choice[np.where(choice == -1)[0]] = 1

            dlen = len(choice)

            # forward step
            # alphas: timesteps x n_states
            alphas = np.ones((dlen, self.n_states))
            rhos = np.zeros((dlen))

            #m_rest = np.sum(self.m_trans, axis=0) / np.sum(self.m_trans)
            alphas[0] = self.m_init * self.m_out[:,choice[0]]
            rhos[0] = np.sum(alphas[0])

            alphas[0] /= rhos[0]

            for i in range(1, dlen):
                # h: n_states
                h = alphas[i-1] @ self.m_trans
                alpha_bar = h * self.m_out[:,choice[i]]
                rhos[i] = np.sum(alpha_bar)
                alphas[i] = alpha_bar / rhos[i]


            # backward step
            # betas: timesteps x n_states
            betas = np.ones((dlen, self.n_states))

            betas[dlen - 1] = np.ones((self.n_states))
            #betas[dlen - 1] /= rhos[-1]

            for i in range(dlen - 1, 0, -1):
                # h: n_states
                h = self.m_out[:,choice[i]] * betas[i]
                beta_bar = self.m_trans @ h
                betas[i - 1] = beta_bar / rhos[i]

            for i in range(len(rhos)):
                if rhos[i] <= 0:
                    pdb.set_trace()
                px[idx] += np.log(rhos[i])

            gammas = np.ones((dlen, self.n_states))
            for i in range(dlen):
                gammas[i] = alphas[i] * betas[i]
                #gammas[i] /= np.sum(gammas[i])

            xi = np.zeros((dlen - 1, self.n_states, self.n_states))

            for t in range(dlen - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t,i,j] = alphas[t,i] * self.m_trans[i,j] * self.m_out[j,choice[t + 1]] * betas[t + 1,j] / rhos[t + 1]

            _gammas += np.sum(gammas, axis=0)
            _gammas_minus_one += np.sum(gammas[:-1,:], axis=0)
            _xi += np.sum(xi, axis=0)
            _inits += gammas[0]
            _gamma_one_counts += np.sum(gammas[np.where(choice == 1)[0],:], axis=0)

        self.m_init = _inits / np.sum(_inits)

        for i in range(self.n_states):
            for j in range(self.n_states):
                self.m_trans[i,j] = _xi[i,j] / _gammas_minus_one[i]

        for i in range(self.n_states):
            self.m_out[i,0] = _gamma_one_counts[i] / _gammas[i]
            self.m_out[i,1] = 1 - self.m_out[i,0]

        total_px = np.sum(px)
        return total_px


    def learn(self, dts, steps=50):

        for i in range(steps):
            px = self._EM_step(dts)
            print(f"step {i}: ll = {px}")
        pdb.set_trace()



if __name__ == '__main__':
    lab = 'hoferlab'
    subject = 'SWC_015'

    one = ONE()

    dts = load_maybe_save(one, lab, subject)
    sessions = [i[3] for i in dts]

    #g = glmhmm(dts)

    g = hmm(states=4)


    g.learn(sessions[1:45])
        
    





