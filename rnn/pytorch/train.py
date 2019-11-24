import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, '../../')

import network, task

def get_default_hp():
    hp = {
        # number of epochs to train
        'n_epochs': 2,
        # learning rate of network
        'learning_rate': 0.01,
        # number of different tasks
        'n_tasks': 3,
        # task id
        'task_id': 1,
        # number of features
        'n_in_features': 10,
        # number of units in the input ring
        'n_in_ring': 0,
        # number of recurrent units
        'n_rec': 30,
        # number of discrete choice output units
        'n_out_choice': 10,
        # number of units in the output ring
        'n_out_ring': 0,
        # activation function
        'activation': torch.tanh,
        # how many steps the RNN takes in total; i.e. how long the trial lasts
        'n_steps': 15,
        # proportion of steps dedicated to fixating at stimulus (task instruction)
        'stim_frac': .2,
        # batch size for training
        'batch_size': 20,
        # residual proportion for RNNs
        'alpha': 0,
        # free steps after fixation before action is required
        'free_steps': 3
    }

    return hp


# generate data loader
def generate_data(hp, samples=3000):

    # first make this simple trial work
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
    n_in_feats = hp['n_in_features']
    n_out_choice = hp['n_out_choice']
    rn = np.random.choice(range(1, n_trials), size=n_in_feats-1, replace=False)
    rn = np.concatenate((np.array([0]), np.sort(rn), np.array([-1])))
    for i in range(n_in_feats):
        # set the stimulus and response values
        v_feat = np.zeros([n_in_feats])
        v_feat[i] = 1
        v_choice = np.zeros([n_out_choice])
        v_choice[i] = 1
        # features stimulus and response values
        trial.put(task_id, ix=[rn[i],rn[i+1],None,trial.n_stim_steps], in_feats=v_feat)
        trial.put(task_id, ix=[rn[i],rn[i+1],trial.n_stim_steps,None], out_choice=v_choice)

    return trial

def train(hp):
    net = network.TaskLSTM(hp)
    fix_criterion = nn.CrossEntropyLoss()
    choice_criterion = nn.MSELoss()
    ring_criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hp['learning_rate'])
    # net_params = [net.W_task]#, net.W_feats, net.W_fix, net.W_rec, net.fix_out_layer.weight, net.choice_out_layer.weight]

    n_stim_steps = int(hp['n_steps'] * hp['stim_frac'])
    fs = hp['free_steps']

    dl = generate_data(hp)
    # flag = 0

    # for the 3 kinds of losses
    train_losses = [[] for l in range(3)]
    # param_vals = [[] for p in range(len(net_params))]
    for epoch in range(hp['n_epochs']):
        train_loss = 0.0
        train_acc = 0.0
        
        net.train()
        
        for i, data in enumerate(dl):
            # inputs; dims (batch_size, n_steps, *)
            x_fix = data['x_fix']
            x_task = data['x_task']
            x_ring = data['x_ring']
            x_feats = data['x_feats']

            # responses: dims (batch_size, n_steps, *)
            y_fix = data['y_fix']
            y_choice = data['y_choice']
            y_ring = data['y_ring']

            # reset the network except for the weights
            net.reset_hidden()
            optimizer.zero_grad()
            
            # X is input to network; dims (n_steps, batch_size, n_input)
            X = torch.cat([x_fix, x_task, x_ring, x_feats], dim=2).transpose(0, 1)

            """
            TODO create a test for this
            if x = X[:,:], then
            x[:1] is fixation
            x[1:1+hp['n_tasks]] is task
            x[1+hp['n_tasks']:1+hp['n_tasks']+hp['n_in_ring']] is x_ring
            x[1+hp['n_tasks']+hp['n_in_ring']:] is x_feats
            """

            rnn_outs = net(X)
            # outs: dims (batch_size, n_steps, *)
            fix_outs, choice_outs, ring_outs = [torch.stack(x).transpose(0, 1) for x in rnn_outs]

            # give fix_outs a free pass for fs timesteps following the fixation stimulus
            # by dissecting fix_outs to become fix_outs_mod
            # fix_outs_spliced, y_fix_spliced; dims(batch_size, n_steps-fs, 1)
            fix_outs_spliced = torch.cat([fix_outs[:,:n_stim_steps,:], fix_outs[:,n_stim_steps+fs:,:]], dim=1)
            y_fix_spliced = torch.cat([y_fix[:,:n_stim_steps,:], y_fix[:,n_stim_steps+fs:,:]], dim=1)

            # fix_outs_mod; dims (batch_size, 2, n_steps-fs)
            # y_fix_mod; dims (batch_size, n_steps-fs)
            fix_outs_mod = torch.cat([fix_outs_spliced, torch.zeros_like(fix_outs_spliced)], dim=2).transpose(1,2)
            y_fix_mod = torch.squeeze(y_fix_spliced.transpose(1,2))

            # calculate the losses
            fix_loss = fix_criterion(fix_outs_mod, y_fix_mod)
            choice_loss = choice_criterion(choice_outs, y_choice)
            ring_loss = ring_criterion(ring_outs, y_ring)

            loss = choice_loss + ring_loss + fix_loss

            # for i,p in enumerate(net_params):
            #     param_vals[i].append(p.detach())
             
            # propagate the losses       
            loss.backward()
            optimizer.step()

            # if type(flag) is int:
            #     flag = np.copy(net.W_task.detach().numpy())


            train_losses[0].append(fix_loss.detach().item())
            train_losses[1].append(choice_loss.detach().item())
            train_losses[2].append(ring_loss.detach().item())
    
    for l in range(3):  
        plt.plot(train_losses[l], label=f'loss type {l}')
    # param_difs = []
    # for i in range(len(param_vals[0]) - 1):
    #     param_difs.append(np.mean((param_vals[0][i+1] - param_vals[0][i]).numpy()))
    # for p in range(len(param_difs)):
    # plt.plot(param_difs)
    plt.legend()
    plt.show()
    pdb.set_trace()




if __name__ == '__main__':
    hp = get_default_hp()
    train(hp)
