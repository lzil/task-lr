import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, '../../')

import network, task

def get_default_hp():
    hp = {
        # number of epochs to train
        'n_epochs': 50,
        # learning rate of network
        'learning_rate': 0.001,
        # number of different tasks
        'n_tasks': 6,
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
        'n_steps': 30,
        # proportion of steps dedicated to fixating at stimulus (task instruction)
        'stim_frac': .4,
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
    rn = np.concatenate((np.array([0]), np.sort(rn), np.array([n_trials])))
    for i in range(n_in_feats):
        # set the stimulus and response values
        v_feat = np.zeros([n_in_feats])
        v_feat[i] = 1
        v_choice = np.zeros([n_out_choice])
        v_choice[i] = 5
        # features stimulus and response values
        trial.put(task_id, ix=[rn[i],rn[i+1],None,trial.n_stim_steps], in_feats=v_feat)
        trial.put(task_id, ix=[rn[i],rn[i+1],trial.n_stim_steps,None], out_choice=v_choice)

    # v_feat = np.zeros([n_in_feats])
    # v_feat[0] = 1
    # v_choice = np.zeros([n_out_choice])
    # v_choice[0] = 0
    # # features stimulus and response values
    # trial.put(task_id, ix=[None,None,None,trial.n_stim_steps], in_feats=v_feat)
    # trial.put(task_id, ix=[None,None,trial.n_stim_steps,None], out_choice=v_choice)

    return trial


def make_graph(hp):
    net = network.TaskLSTM(hp)
    writer = SummaryWriter()
    dl = generate_data(hp)

    # get some random training images
    dataiter = iter(dl)
    data = dataiter.next()

    x_fix = data['x_fix']
    x_task = data['x_task']
    x_ring = data['x_ring']
    x_feats = data['x_feats']

    # responses: dims (batch_size, n_steps, *)
    y_fix = data['y_fix']
    y_choice = data['y_choice']
    y_ring = data['y_ring']
    
    # X is input to network; dims (batch_size, n_steps, n_input)
    X = torch.cat([x_fix, x_task, x_ring, x_feats], dim=2)

    # create grid of images

    writer.add_graph(net, X)

def train(hp):
    net = network.TaskGRU(hp)

    writer = SummaryWriter()

    fix_criterion = nn.CrossEntropyLoss()
    choice_criterion = nn.MSELoss()
    ring_criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hp['learning_rate'])
    # net_params = [net.W_task]#, net.W_feats, net.W_fix, net.W_rec, net.fix_out_layer.weight, net.choice_out_layer.weight]

    n_stim_steps = int(hp['n_steps'] * hp['stim_frac'])
    fs = hp['free_steps']

    dl = generate_data(hp)

    # for the 3 kinds of losses
    train_losses = [[] for l in range(3)]


    g_step = 0
    for epoch in range(hp['n_epochs']):
        train_loss = 0.0
        train_acc = 0.0
        
        net.train()
        
        for i, data in enumerate(dl):
            g_step += 1
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
            
            # X is input to network; dims (batch_size, n_steps, n_input)
            X = torch.cat([x_fix, x_task, x_ring, x_feats], dim=2)
            
            # Y is correct response to network; dims (batch_size, n_steps, n_input)
            Y = torch.cat([y_fix, y_choice, y_ring], dim=2)

            """
            TODO create a test for this
            if x = X[:,:], then
            x[:1] is fixation
            x[1:1+hp['n_tasks]] is task
            x[1+hp['n_tasks']:1+hp['n_tasks']+hp['n_in_ring']] is x_ring
            x[1+hp['n_tasks']+hp['n_in_ring']:] is x_feats
            """

            # outs: dims (batch_size, n_steps, *)
            fix_outs, choice_outs, ring_outs = net(X)
            Z = torch.cat([fix_outs, choice_outs, ring_outs], dim=2)

            # give fix_outs a free pass for fs timesteps following the fixation stimulus
            # by dissecting fix_outs to become fix_outs_mod
            # fix_outs_spliced, y_fix_spliced; dims(batch_size, n_steps-fs, 1)
            fix_outs_spliced = torch.cat([fix_outs[:,:n_stim_steps,:], fix_outs[:,n_stim_steps+fs:,:]], dim=1)
            y_fix_spliced = torch.cat([y_fix[:,:n_stim_steps,:], y_fix[:,n_stim_steps+fs:,:]], dim=1)

            # fix_outs_mod; dims (batch_size, 2, n_steps-fs)
            # y_fix_mod; dims (batch_size, n_steps-fs)
            # fix_outs_mod = torch.cat([fix_outs_spliced, torch.zeros_like(fix_outs_spliced)], dim=2).transpose(1,2)
            fix_outs_trans = fix_outs_spliced.transpose(1,2)
            fix_outs_opp = torch.zeros_like(fix_outs_trans, requires_grad=False)
            fix_outs_mod = torch.cat([fix_outs_trans,fix_outs_opp], dim=1)
            y_fix_mod = torch.squeeze(y_fix_spliced.transpose(1,2)).to(torch.int64)

            # calculate the losses
            fix_loss = fix_criterion(fix_outs_mod, y_fix_mod)
            choice_loss = choice_criterion(choice_outs, y_choice)
            ring_loss = ring_criterion(ring_outs, y_ring)

            loss = choice_loss + ring_loss + fix_loss
             
            # propagate the losses       
            loss.backward()
            optimizer.step()

            for idx, p in enumerate(net.params):
                # writer.add_scalar(f'grad_{idx}', torch.mean(p.grad), g_step)
                # writer.add_scalar(f'param_{idx}', torch.mean(torch.abs(p)), g_step)
                writer.add_histogram(f'W_{idx}', p, g_step)

            train_losses[0].append(fix_loss.detach().item())
            train_losses[1].append(choice_loss.detach().item())
            train_losses[2].append(ring_loss.detach().item())

            for idx in range(2):
                writer.add_scalar(f'loss_{idx}', train_losses[idx][-1], g_step)

            if g_step % 50 == 0:
                
                writer.add_image('stimulus', X[0,:,:], g_step, dataformats='HW')
                writer.add_image('response', Z[0,:,:], g_step, dataformats='HW')
                writer.add_image('Y', Y[0,:,:], g_step, dataformats='HW')

    writer.close()
    
    # for l in range(3):  
    #     plt.plot(train_losses[l], label=f'loss type {l}')
    # param_difs = []
    # for i in range(len(param_vals[0]) - 1):
    #     param_difs.append(np.mean((param_vals[0][i+1] - param_vals[0][i]).numpy()))
    # for p in range(len(param_difs)):
    # plt.plot(param_difs)
    # plt.legend()
    # plt.show()
    #pdb.set_trace()




if __name__ == '__main__':
    hp = get_default_hp()
    train(hp)
