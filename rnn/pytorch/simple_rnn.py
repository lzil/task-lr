import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


def default_for(var, val):
    if var is not None:
        return var
    return val



class SimpleRNN(nn.Module):
    def __init__(self, n_input, n_rec, n_output, activation=torch.tanh):
        super().__init__()

        self.n_input = n_input
        self.n_rec = n_rec
        self.n_output = n_output

        # random initial weight
        self.W_input = torch.nn.Parameter(torch.randn(n_input, n_rec))
        self.W_rec = torch.nn.Parameter(torch.randn(n_rec, n_rec))
        self.b_rec = torch.nn.Parameter(torch.zeros(1, n_rec))
        
        self.W_output = torch.nn.Parameter(torch.randn(n_rec, n_output))
        self.b_out = torch.nn.Parameter(torch.zeros(1, n_output))

        self.activation = activation

        self.h = torch.zeros(1, n_rec)
        
    def reset_hidden(self):
        self.h = torch.zeros(1, self.n_rec)


    def forward(self, X):
        # X should be of size (1, n_input)
        self.h = self.activation(
            torch.mm(X, self.W_input) + \
            torch.mm(self.h, self.W_rec) + \
            self.b_rec)

        self.out = self.activation(
            torch.mm(self.h, self.W_output) + \
            self.b_out)

        return self.h, self.out

class CleanRNN(nn.Module):
    def __init__(self, hp):
        super().__init__()

        self.n_input = hp['num_input']
        self.n_rec = hp['num_rec']
        self.n_output = hp['num_output']

        self.n_steps = hp['run_steps']
        self.activation = default_for(hp['activation'], 'tanh')

        self.batch_size = hp['batch_size']

        self.rnn_cell = torch.nn.RNNCell(
            input_size=self.n_input,
            hidden_size=self.n_rec,
            bias=True,
            nonlinearity=self.activation
            )

        self.output_layer = torch.nn.Linear(
            in_features=self.n_rec,
            out_features=self.n_output,
            bias=True
            )

        self.rnn_rec = torch.randn(self.batch_size, self.n_rec)

    def reset_hidden(self):
        self.rnn_rec = torch.randn(self.batch_size, self.n_rec)

    def forward(self, X):
        # X is input, dims (num_steps, batch_size, num_rec)

        self.rnn_recs = []
        self.rnn_outs = []
        for in_step in X:
            self.rnn_rec = self.rnn_cell(in_step, self.rnn_rec)
            self.rnn_out = self.output_layer(self.rnn_rec)
            self.rnn_recs.append(self.rnn_rec)
            self.rnn_outs.append(self.rnn_out)

        return self.rnn_outs



class TaskDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {'x': torch.from_numpy(self.x[idx]).to(torch.float),
                  'y': torch.from_numpy(self.y[idx]).to(torch.float)
                 }
            
        return sample


def get_default_hp():
    hp = {
        # number of epochs to train
        'num_epochs': 2,
        # learning rate of network
        'learning_rate': 0.01,
        # number of input units
        'num_input': 4,
        # number of recurrent units
        'num_rec': 10,
        # number of output units
        'num_output': 4,
        # activation function
        'activation': 'tanh',
        # how many steps the RNN takes in total; i.e. how long the trial lasts
        'run_steps': 50,
        # proportion of steps dedicated to fixating at stimulus (task instruction)
        'stim_frac': .2,
        # batch size for training
        'batch_size': 2
    }

    return hp

def generate_data(hp, samples=20):
    stim_steps = int(hp['run_steps'] * hp['stim_frac'])
    x_stim = np.zeros((samples, stim_steps, hp['num_input']), dtype=np.float)
    y_stim = np.copy(x_stim)
    x_go = np.zeros((samples, hp['run_steps'] - stim_steps, hp['num_input']), dtype=np.float)
    y_go = np.copy(x_go)

    # set the stimulus values, and correct response values
    x_stim[:,:,[0,1]] = 1
    y_stim[:,:,0] = 1
    y_go[:,:,1] = 1

    # combine stimulus and responses
    x_all = np.concatenate((x_stim, x_go), axis=1)
    y_all = np.concatenate((y_stim, y_go), axis=1)

    td = TaskDataset(x_all, y_all)
    dl = DataLoader(
        dataset=td,
        batch_size=hp['batch_size'],
        shuffle=True,
        drop_last=True
        )

    return dl


def train(hp):
    net = CleanRNN(hp)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hp['learning_rate'])

    dl = generate_data(hp)

    train_losses = []
    for epoch in range(hp['num_epochs']):
        train_loss = 0.0
        train_acc = 0.0
        
        net.train()
        
        for i,data in enumerate(dl):
            x_cur = data['x'].transpose(0,1)
            y_cur = data['y'].transpose(0,1)

            
            net.reset_hidden()
            optimizer.zero_grad()
            
            outs = net(x_cur)
            out_tensor = torch.stack(outs)
            loss = criterion(out_tensor,y_cur)
                    
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())
            
    plt.plot(train_losses)
    plt.show()




if __name__ == '__main__':
    hp = get_default_hp()
    train(hp)




