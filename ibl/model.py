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
    def __init__(self, dts, states=2):

        self.n_states = states

        self.dts = [i[3] for i in dts]

        self.dt = self.dts[56]

        self.choice = self.dt.choice
        self.choice[np.where(self.choice == 1)[0]] = 0
        self.choice[np.where(self.choice == -1)[0]] = 1

        # normalized transition matrix
        # starting state x ending state
        m_trans = np.random.rand(self.n_states, self.n_states)
        self.m_trans = (m_trans / np.sum(m_trans, axis=0)).T
        print(self.m_trans)

        # output matrix
        m_out = np.random.rand(self.n_states, 2)
        self.m_out = (m_out.T / np.sum(m_out, axis=1)).T
        print(self.m_out)

        m_init = np.random.rand(self.n_states)
        self.m_init = m_init / np.sum(m_init)
    



    def forward(self):
        choice = self.choice
        dlen = len(choice)

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

        return alphas, rhos

    def backward(self, rhos):
        choice = self.choice
        dlen = len(choice)

        # betas: timesteps x n_states
        betas = np.ones((dlen, self.n_states))

        betas[dlen - 1] = np.ones((self.n_states))
        #betas[dlen - 1] /= rhos[-1]

        for i in range(dlen - 1, 0, -1):
            # h: n_states
            h = self.m_out[:,choice[i]] * betas[i]
            beta_bar = self.m_trans @ h
            betas[i - 1] = beta_bar / rhos[i]

        return betas


    def EM_step(self):
        choice = self.choice
        dlen = len(choice)

        alphas, rhos = self.forward()
        betas = self.backward(rhos)

        Px = 1
        for i in range(len(rhos)):
            Px += np.log(rhos[i])

        gammas = np.ones((dlen, self.n_states))
        for i in range(dlen):
            gammas[i] = alphas[i] * betas[i]
            #gammas[i] /= np.sum(gammas[i])

        #Px = alphas[0,:] @ betas[0,:]

        xi = np.zeros((dlen - 1, self.n_states, self.n_states))

        for t in range(dlen - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t,i,j] = alphas[t,i] * self.m_trans[i,j] * self.m_out[j,choice[t + 1]] * betas[t + 1,j] / rhos[t + 1]


        #self.m_init = np.mean(gammas, axis=0)


        for i in range(self.n_states):
            for j in range(self.n_states):
                self.m_trans[i,j] = np.sum(xi[:,i,j]) / np.sum(gammas[:-1,i])

        for i in range(self.n_states):
            self.m_out[i,0] = np.sum(gammas[np.where(choice == 1)[0],i]) / np.sum(gammas[:,i])
            self.m_out[i,1] = 1 - self.m_out[i,0]


        print('Px', Px)
        return Px




if __name__ == '__main__':
    lab = 'hoferlab'
    subject = 'SWC_015'

    one = ONE()

    dts = load_maybe_save(one, lab, subject)

    #g = glmhmm(dts)

    g = hmm(dts)

    for i in range(100):
        Pd = g.EM_step()
    pdb.set_trace()
        
    





