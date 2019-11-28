import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from PIL import Image

from torch.utils.data import Dataset, DataLoader

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import time

sys.path.insert(1, '../../')

from network import *
from trial import *

from tools import *
from dataset import *

program_mode = 'train'
verb = 'true'


def get_default_hp():
    hp = {
        # number of epochs to train
        'n_epochs': 600,
        # batch size for training
        'batch_size': 20,
        # learning rate of network
        'learning_rate': 0.005,
        # number of different tasks
        'n_tasks': 6,
        # number of features
        'n_in_features': 0,
        # number of units in the input ring
        'n_in_ring': 24,
        # number of recurrent units
        'n_rec': 50,
        # number of discrete choice output units
        'n_out_choice': 0,
        # number of units in the output ring
        'n_out_ring': 24,
        # activation function
        'activation': 'tanh',
        # how many steps the RNN takes in total; i.e. how long the trial lasts
        'n_steps': 30,
        # rnn type
        'rnn_type': 'gru',
        # residual proportion for RNNs
        'alpha': 0.2,
        # recurrent noise in RNNs,
        'sigma_rec': 0.05,
        # input noise in RNNs
        'sigma_x': 0.01,
    }

    return hp


def get_loss_weights(hp, Y):
    # cost masks
    loss_weights = {}

    n_input = 1 + hp['n_tasks'] + hp['n_in_features'] + hp['n_in_ring']
    n_output = 1 + hp['n_out_choice'] + hp['n_out_ring']

    batch_size = hp['batch_size']
    n_steps = hp['n_steps']

    # cross entropy loss at every timestep
    out_total = torch.ones((batch_size, n_steps, n_output))

    one_weights = torch.ones((batch_size, n_steps))
    zero_locs = torch.max(Y, dim=2)[0] != 0
    loss_weights['one'] = one_weights * zero_locs
    loss_weights['saccade'] = 1

    # L1 loss
    loss_weights['l1'] = 1e-6

    # L2 loss
    
    #loss_weights['l2'] = 1e-6
    #Z = torch.ones((batch_size, n_steps, ))

    return loss_weights

def SaccadeLoss(ring_buf=0):
    # z should be (batch_size, n_steps, n_output)
    def loss(z):
        zm = torch.argmax(z, dim=2)
        # TODO write simpler code using ring buffer sometime
        # r_start = 1 + hp['n_tasks'] + hp['n_out_choice']
        # r_end = ring_start + hp['n_out_ring']
        # if ring_buf > 0:
        #     dif = torch.abs(zm[i] - zm[i-1])
        #     n = sum([
        #         torch.abs(zm[i] - zm[i-1]) > 1 and (
        #             not_in_ring(zm[i]) or not_in_ring(zm[i-1]) or
        #             not (torch.abs(zm[i] - zm[i-1]) % hp['n_out_ring'])
        #             )
        #         for i in range(1,zm.numel())])

        ns = [zm[:,i] != zm[:,i-1] for i in range(1,zm.size()[1])]
        total = torch.stack(ns, dim=1).float() / zm.size()[1]
        return total
    return loss

def train(
    hp=None,
    run_path=None,
    data_path=None,
    model_path=None
    ):
    
    if hp == None:
        print('Hyperparameters not specified; running with defaults.')
        hp = get_default_hp()

    rnnt = hp['rnn_type']
    if rnnt == 'rnn':
        net = TaskRNN(hp)
    elif rnnt == 'gru':
        net = TaskGRU(hp)
    elif rnnt == 'lstm':
        net = TaskLSTM(hp)

    print(f'Net ({rnnt}) initialized.')

    writer = SummaryWriter()

    print(f'Summary writer initialized.')

    optimizer = optim.Adam(net.parameters(), lr=hp['learning_rate'])

    #n_stim_steps = int(hp['n_steps'] * hp['fix_frac'])

    dl = get_data_loader(hp, data_path)
    if model_path is not None:
        ckpt = torch.load(mp)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        print('Starting from previous train state.')


    # loss criteria and variables
    # fix_criterion = nn.BCEWithLogitsLoss(reduction='none')
    # choice_criterion = nn.BCEWithLogitsLoss(reduction='none')
    # ring_criterion = nn.BCEWithLogitsLoss(reduction='none')

    # weight cost mask not implementable in get_loss_weights
    n_output = 1 + hp['n_out_choice'] + hp['n_out_ring']
    cost_mask = torch.ones(n_output)
    cost_mask[0] = 5
    one_criterion = nn.CrossEntropyLoss(weight=cost_mask, reduction='none')
    saccade_criterion = SaccadeLoss()
    l1_criterion = nn.L1Loss()

    loss_types = ['one', 'l1', 'saccade']

    losses = []
    losses_weighted = {key:[] for key in loss_types}

    print(f'Training a ({rnnt}):\n\t \
        epochs: {hp["n_epochs"]}\n\t \
        samples: {len(dl) * hp["batch_size"]}\n\t \
        batch_size: {hp["batch_size"]}\
        ')

    g_step = 0
    for epoch in range(hp['n_epochs']):
        
        net.train()
        for i, data in enumerate(dl):
            g_step += 1
            losses_unweighted = {}

            # reset the network except for the weights
            net.reset_hidden()
            optimizer.zero_grad()

            # inputs; dims (batch_size, n_steps, *)
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
            
            # Y is correct response to network; dims (batch_size, n_steps, n_output)
            Y = torch.cat([y_fix, y_choice, y_ring], dim=2)
            Y_C = torch.argmax(Y, dim=2)

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
            if (Z != Z).any():
                pdb.set_trace()
            

            # l1 regularization
            l1 = 0
            for p in net.parameters():
                l1 += p.abs().sum()
            losses_unweighted['l1'] = l1
            # cross entropy loss across each timestep
            losses_unweighted['one'] = one_criterion(Z.transpose(1,2), Y_C)
            # saccading is expensive
            losses_unweighted['saccade'] = saccade_criterion(Z)
            

            # weight the losses and sum them to form the actual loss
            loss = 0
            loss_weights = get_loss_weights(hp, Y)
            for l in loss_types:
                if losses_unweighted[l].numel() > 0:
                    val = torch.mean(losses_unweighted[l] * loss_weights[l])
                    losses_weighted[l].append(val.detach().item())
                    loss += val
                else:
                    losses_weighted[l].append(0)

            losses.append(loss.detach().item())
            
            # propagate the losses       
            loss.backward()
            optimizer.step()

            # add losses to tensorboard
            for l in loss_types:
                writer.add_scalar(f'loss_{l}', losses_weighted[l][-1], g_step)
            writer.add_scalar('total loss', loss, g_step)

            if g_step % 200 == 0:
                # add weights to tensorboard
                for k,v in net.state_dict().items():
                    if v.numel() != 0:
                        writer.add_histogram(k, v, g_step)
                # add simple visualization to tensorboard
                vs = [v[0].detach().numpy() for v in [X,Y,Z]]
                vs[0] = vs[0].clip(0,1)
                vs[1] = vs[1].clip(0,1)
                vs[2] = sps.softmax(vs[2],axis=1)
                vs_np = [(255 * v).astype(np.uint8) for v in vs]
                fig, ax = plt.subplots(nrows=1,ncols=3)
                for v in range(3):
                    img = Image.fromarray(vs_np[v])
                    ax[v].imshow(img, cmap='gray')
                    ax[v].set_title(f'{v+1}')
                writer.add_figure('X, Y, Z', fig, g_step)

        avg_loss = np.mean(np.array(losses[-80:]))
        print(f'Epoch {epoch+1} completed. Loss: {avg_loss}')
        if epoch % 2 == 0 and program_mode != 'debug' and run_path is not None:
            # save checkpoint
            ckpt_path = os.path.join(run_path, str(g_step)+'.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)
            print(f'Checkpoint saved at {ckpt_path}.')

    writer.close()

    print(f'Finished running.\n\tSteps:{g_step}')


    pdb.set_trace()



if __name__ == '__main__':

    cur_time = time.ctime().replace(' ', '_')
    run_path = os.path.join('models', cur_time)

    data_path='data/Thu_Nov_28_00:50:15_2019/match_ring+average_ring.pkl'
    model_path=None #'models/Tue_Nov_26_00:31:21_2019/7000.tar'

    if program_mode == 'train':
        mkdir_p(run_path)
        hp = get_default_hp()

    if program_mode == 'debug':
        hp = get_default_hp()
        hp['n_epochs'] = 1
        hp['n_samples'] = 500
        hp['batch_size'] = 5

    train(hp,
        run_path,
        data_path,
        model_path
    )



