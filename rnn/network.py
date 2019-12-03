import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributions as distributions

import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from abc import ABC, abstractmethod

import tools

class TaskNet(nn.Module, ABC):
    def __init__(self, hp):
        super().__init__()

        self.n_task = hp['n_tasks']
        self.n_in_feats = hp['n_in_features']
        self.n_in_ring = hp['n_in_ring']
        self.n_input = 1 + self.n_task + self.n_in_feats + self.n_in_ring
        self.n_rec = hp['n_rec']
        self.n_out_choice = hp['n_out_choice']
        self.n_out_ring = hp['n_out_ring']

        self.n_steps = hp['n_steps']
        self.batch_size = hp['batch_size']

        if hp['activation'] == 'tanh':
            self.activation = torch.tanh
        elif hp['activation'] == 'relu':
            self.activation = torch.ReLU
        elif hp['activation'] == 'softplus':
            self.activation = torch.nn.Softplus()

        self.alpha = hp['alpha']

        self.sigma_rec = hp['sigma_rec']
        self.noise_const = np.sqrt(2 / self.alpha) * self.sigma_rec
        self.randn_gen = distributions.Normal(0, 1)

        # fixation output
        self.fix_out_layer = torch.nn.Linear(
            in_features=self.n_rec,
            out_features=1,
            bias=True
            )

        # choices output
        self.choice_out_layer = torch.nn.Linear(
            in_features=self.n_rec,
            out_features=self.n_out_choice,
            bias=True
            )

        # ring output
        self.ring_out_layer = torch.nn.Linear(
            in_features=self.n_rec,
            out_features=self.n_out_ring,
            bias=True
            )


        # hidden state; dims (batch_size, n_rec)
        self.state = torch.randn(*(self.batch_size, self.n_rec))

    # implementation of recurrent computation
    @abstractmethod
    def _cell(self, inp):
        pass

    # reset hidden state between batches
    @abstractmethod
    def reset_hidden(self):
        pass

    # X is input; dims (batch_size, n_steps, n_input) if batch_first is true
    def forward(self, X, batch_first=True):
        if batch_first:
            X = X.transpose(0, 1)
        rnn_outs = [[] for i in range(3)]
        for in_step in X:
            # recurrent step
            state = self._cell(in_step)

            # rnn outs; dims (batch_size, *)
            # magnitude of fixation choice
            rnn_fix = self.fix_out_layer(state)
            # normal linear layer choices. TODO maybe use sigmoid instead cus it's just attention? think about it
            rnn_choice = self.choice_out_layer(state)
            rnn_ring = self.ring_out_layer(state)

            # rnn_out = torch.cat([rnn_fix, rnn_choice, rnn_ring], dim=2)

            rnn_outs[0].append(rnn_fix)
            rnn_outs[1].append(rnn_choice)
            rnn_outs[2].append(rnn_ring)

        # after transpose, each rnn_out is (batch_size, n_steps, *)
        rnn_outs = [torch.stack(out,dim=0).transpose(0,1) for out in rnn_outs]
        return tuple(rnn_outs)



# vanilla RNN
class TaskRNN(TaskNet):
    def __init__(self, hp):
        super().__init__(hp)

        # input weights
        self.W_in = nn.Parameter(torch.randn(self.n_input, self.n_rec))

        # recurrent weights
        self.W_rec = torch.nn.Parameter(torch.randn(self.n_rec, self.n_rec))
        self.b_rec = torch.nn.Parameter(torch.randn(1, self.n_rec))

    # custom recurrent cell code, could just use torch.nn.RNNCell
    def _cell(self, inp):
        noise = self.noise_const * self.randn_gen.rsample((self.batch_size, self.n_rec))
        state = self.activation(
            torch.mm(inp, self.W_in) +
            torch.mm(self.state, self.W_rec) +
            self.b_rec +
            noise)

        # adding noise

        # masking
        self.state = self.alpha * self.state + (1-self.alpha) * state
        return self.state

    def reset_hidden(self):
        self.state = torch.randn(*(self.batch_size, self.n_rec))



    

# LSTM
class TaskLSTM(TaskNet):
    def __init__(self, hp):
        super().__init__(hp)


        # gates
        self.W_f = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))
        self.W_i = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))
        self.W_C = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))
        self.W_o = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))

        # gate biases
        self.b_f = nn.Parameter(torch.randn(1, self.n_rec))
        self.b_i = nn.Parameter(torch.randn(1, self.n_rec))
        self.b_C = nn.Parameter(torch.randn(1, self.n_rec))
        self.b_o = nn.Parameter(torch.randn(1, self.n_rec))

        # cell state present in LSTMs
        self.C = torch.randn(*(self.batch_size, self.n_rec))


    def _cell(self, inp):
        # from: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

        # cat is (batch_size, n_rec + n_input)
        cat = torch.cat([self.state, inp], dim=1)
        # forget gate: f is (batch_size, n_rec)
        f = torch.sigmoid(torch.mm(cat, self.W_f) + self.b_f) 
        # input gate: i is (batch_size, n_rec)
        i = torch.sigmoid(torch.mm(cat, self.W_i) + self.b_i)
        # candidate values: C_tilde is (batch_size, n_rec)
        C_tilde = torch.tanh(torch.mm(cat, self.W_C) + self.b_C)
        # cell state update: C is (batch_size, n_rec)
        self.C = f * self.C + i * C_tilde
        # output gate: o is (batch_size, n_rec)
        o = torch.sigmoid(torch.mm(cat, self.W_o) + self.b_o)
        # hidden state: state is (batch_size, n_rec)
        self.state = o * torch.tanh(self.C)

        return self.state

    def reset_hidden(self):
        self.state = torch.randn(*(self.batch_size, self.n_rec))
        self.C = torch.randn(*(self.batch_size, self.n_rec))




    
# GRU
class TaskGRU(TaskNet):
    def __init__(self, hp):
        super().__init__(hp)

        # gates and weights
        self.W_z = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))
        self.W_r = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))
        self.W = nn.Parameter(torch.randn(self.n_input + self.n_rec, self.n_rec))

        # gate biases
        self.b_z = nn.Parameter(torch.randn(1, self.n_rec))
        self.b_r = nn.Parameter(torch.randn(1, self.n_rec))
        self.b = nn.Parameter(torch.randn(1, self.n_rec))
        

    def _cell(self, inp):
        # from: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

        # cat, r_cat; dims (batch_size, n_rec + n_input)
        cat = torch.cat([self.state, inp], dim=1)
        # z, r, h_tilde; dims (batch_size, n_rec)
        z = torch.sigmoid(torch.mm(cat, self.W_z) + self.b_z)
        r = torch.sigmoid(torch.mm(cat, self.W_r) + self.b_r)
        rcat = torch.cat([r * self.state, inp], dim=1)

        noise = self.noise_const * self.randn_gen.rsample((self.batch_size, self.n_rec))
        h_tilde = torch.tanh(torch.mm(rcat, self.W) + self.b + noise)

        # hidden state; dims (batch_size, n_rec)
        self.state = (1 - self.alpha * z) * self.state + self.alpha * z * h_tilde
        return self.state


    def reset_hidden(self):
        self.state = torch.randn(*(self.batch_size, self.n_rec))





        



