import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import task


def default_for(var, val):
    if var is not None:
        return var
    return val

class CleanRNN(nn.Module):
    def __init__(self, hp):
        super().__init__()

        self.n_feats = hp['n_features']
        self.n_task = hp['n_tasks']
        self.n_ring = hp['n_ring']

        self.n_rec = hp['n_rec']
        self.n_output = hp['n_output']
        self.n_steps = hp['n_steps']
        self.activation = default_for(hp['activation'], torch.nn.Tanh())

        self.batch_size = hp['batch_size']

        # input weights, separated so it is easy to look at individual weights
        self.W_fix = torch.nn.Parameter(torch.randn(1, self.n_rec))
        self.W_ring = torch.nn.Parameter(torch.randn(self.n_ring, self.n_rec))
        self.W_task = torch.nn.Parameter(torch.randn(self.n_task, self.n_rec))
        self.W_feats = torch.nn.Parameter(torch.randn(self.n_feats, self.n_rec))

        # combination of the above weights into one input tensor
        self.W_in = torch.cat([self.W_fix, self.W_task, self.W_ring, self.W_feats])

        # recurrent weights
        self.W_rec = torch.nn.Parameter(torch.randn(self.n_rec, self.n_rec))
        self.b_rec = torch.nn.Parameter(torch.randn(1, self.n_rec))

        # output layer
        self.output_layer = torch.nn.Linear(
            in_features=self.n_rec,
            out_features=self.n_output,
            bias=True
            )

        # output fixation
        self.output_fix_layer = torch.nn.Linear(
            in_features=self.n_rec,
            out_features=1
            )

        # hidden state
        self.rnn_rec = torch.randn(self.batch_size, self.n_rec)

    # custom recurrent cell code, could just use torch.nn.RNNCell
    def rec_cell(self, inp):
        rnn_rec = self.activation(
            torch.mm(inp, self.W_in) +
            torch.mm(self.rnn_rec, self.W_rec) +
            self.b_rec)
        return rnn_rec

    # reset hidden state between batches
    def reset_hidden(self):
        self.rnn_rec = torch.randn(self.batch_size, self.n_rec)

    # run the RNN with a batch
    def forward(self, x_fix, x_task, x_ring, x_feats):
        # X is input, dims (n_steps, batch_size, n_input)
        X = torch.cat([x_fix, x_task, x_ring, x_feats], dim=2).transpose(0, 1)

        rnn_recs = []
        rnn_outs = []
        rnn_fix_outs = []
        for in_step in X:
            # recurrent step
            self.rnn_rec = self.rec_cell(in_step)
            # normal linear layer output
            rnn_out = self.output_layer(self.rnn_rec)
            # output of fixation is in [0,1]
            rnn_fix_out = torch.nn.Sigmoid()(self.output_fix_layer(self.rnn_rec))
            rnn_recs.append(self.rnn_rec)
            rnn_outs.append(rnn_out)
            rnn_fix_outs.append(rnn_fix_out)

        return rnn_outs, rnn_fix_outs


def get_default_hp():
    hp = {
        # number of epochs to train
        'n_epochs': 2,
        # learning rate of network
        'learning_rate': 0.0005,
        # number of different tasks
        'n_tasks': 3,
        # task id
        'task_id': 1,
        # number of features
        'n_features': 10,
        # number of units in the ring
        'n_ring': 8,
        # number of recurrent units
        'n_rec': 30,
        # number of output units
        'n_output': 10,
        # activation function
        'activation': torch.nn.Tanh(),
        # how many steps the RNN takes in total; i.e. how long the trial lasts
        'n_steps': 15,
        # proportion of steps dedicated to fixating at stimulus (task instruction)
        'stim_frac': .2,
        # batch size for training
        'batch_size': 10
    }

    return hp

# generate data loader
def generate_data(hp, samples=3000):

    trial = make_delay_match_trial(hp, samples)

    td = task.TrialData([trial])
    dl = DataLoader(
        dataset=td,
        batch_size=hp['batch_size'],
        shuffle=True,
        drop_last=True
        )

    return dl

# test for delay match task
def make_delay_match_trial(hp, n_trials):

    trial = task.Trial(hp, n_trials=n_trials)

    task_id = hp['task_id']
    n_feats = hp['n_features']
    n_output = hp['n_output']
    rn = np.random.choice(range(1, n_trials), size=n_feats, replace=False)
    rn = np.concatenate((np.array([0]), np.sort(rn), np.array([-1])))
    for i in range(n_feats):
        # set the stimulus and response values
        v_feat = np.zeros([n_feats])
        v_feat[i] = 1
        v_resp = np.zeros([n_output])
        v_resp[i] = 1
        # features stimulus and response values
        trial.put(task_id, ix=[rn[i],rn[i+1],None,trial.n_stim_steps], feats=v_feat)
        trial.put(task_id, ix=[rn[i],rn[i+1],trial.n_stim_steps,None], resp=v_resp)

    return trial


def train(hp):
    net = CleanRNN(hp)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hp['learning_rate'])

    dl = generate_data(hp)

    train_losses = []
    for epoch in range(hp['n_epochs']):
        train_loss = 0.0
        train_acc = 0.0
        
        net.train()
        
        for i, data in enumerate(dl):
            x_fix = data['x_fix']
            x_task = data['x_task']
            x_ring = data['x_ring']
            x_feats = data['x_feats']
            y_fix = data['y_fix']
            y_resp = data['y_resp']

            
            net.reset_hidden()
            optimizer.zero_grad()
            
            outs, fix_outs = net(x_fix, x_task, x_ring, x_feats)
            outs_tensor = torch.cat([torch.stack(fix_outs), torch.stack(outs)], dim=2).transpose(0, 1)
            y_tensor = torch.cat([y_fix, y_resp], dim=2)


            loss = criterion(outs_tensor,y_tensor)
                    
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())
            
    plt.plot(train_losses)
    plt.show()




if __name__ == '__main__':
    hp = get_default_hp()
    train(hp)


