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

states = 3


# cues: contrast, prev choices, feedback



def calculate_nll(dt):
    state = 0
    ll = 0


class glmhmm:
    def __init__(self, dts):
        self.dts = [dts[1] for i in dts]



class hmm:
    def __init__(self, dts, states=3):

        self.n_states = states

        self.dts = [i[2] for i in dts]

        self.dt = self.dts[0]

        # normalized transition matrix
        # starting state x ending state
        m_trans = np.random.rand(3,3)
        self.m_trans = m_trans / np.sum(m_trans, axis=1)

        # output matrix
        self.m_out = np.ones((states, 2)) / 2
    



    def forward(self):
        choice = self.dt.choice
        dlen = len(choice)

        # alphas: timesteps x n_states
        alphas = np.ones((dlen, states))
        rhos = np.zeros((dlen))

        m_rest = np.sum(self.m_trans, axis=0) / np.sum(self.m_trans)
        alphas[0] = (m_rest @ self.m_out)[choice[0]]
        rho[0] = np.sum(alphas[0])

        for i in range(1, dlen):
            # h: n_states
            h = alphas[i-1] @ self.m_trans
            alpha_bar = (h * self.m_out[:,choice[i]])
            rho[i] = np.sum(alpha_bar)
            alphas[i] = alpha_bar / rho[i]

        return alphas

    def backward(self):
        choice = self.dt.choice
        dlen = len(choice)

        # betas: timesteps x n_states
        betas = np.ones((dlen, states))

        betas[dlen - 1] = np.ones((states))

        for i in range(dlen - 1, 0, -1):
            # h: n_states
            h = self.m_out[:,choice[i]] * betas[i]
            betas[i - 1] = self.m_trans @ h

        return betas


    def EM_step(self):
        choice = self.dt.choice
        dlen = len(choice)

        alphas = self.forward()
        betas = self.backward()

        gammas = np.ones((dlen, states))
        for i in range(dlen):
            gammas[i] = sps.softmax(alphas[i] * betas[i])

        Pd = alphas[0,:] @ betas[0,:]
        xi = np.zeros((dlen - 1, states, states))

        for i in range(dlen - 1):
            for j in range(states):
                for k in range(states):
                    xi[i,j,k] = alphas[i,j] * self.m_trans[j,k] * self.m_out[k,choice[i]] * betas[i + 1,k]

        xi /= Pd

        for i in range(states):
            for j in range(states):
                self.m_trans[i,j] = np.sum(xi[:,i,j]) / np.sum(gammas[:-1,i])

        for i in range(states):
            self.m_out[i,0] = np.sum(gammas[np.where(choice == 1)[0],i]) / np.sum(gammas[:,i])
            self.m_out[i,1] = np.sum(gammas[np.where(choice == -1)[0],i]) / np.sum(gammas[:,i])


        return Pd




if __name__ == '__main__':
    lab = 'hoferlab'
    subject = 'SWC_015'

    one = ONE()

    dts = load_maybe_save(one, lab, subject)

    #g = glmhmm(dts)

    g = hmm(dts)

    for i in range(10):
        Pd = g.EM_step()
        pdb.set_trace()
        print(Pd)





