import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

import pickle

from torch.utils.data import Dataset, DataLoader

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import time

sys.path.insert(1, '../')

from network import *
from trial import *
from tools import *
from train import *


# generate data loader
def get_data_loader(hp, data_path=None):

    if data_path is not None:
        print('Loading dataset...')
        with open(data_path, 'rb') as f:
            td = pickle.load(f)
        print('Dataset loaded.')
    else:
        td = create_dataset(hp)

    dl = DataLoader(
        dataset=td,
        batch_size=hp['batch_size'],
        shuffle=True,
        drop_last=True
        )
    

    return dl


def create_dataset(hp, dir_path=None):
    print('Creating dataset...')

    trials = []

    # list the trials you want to do here
    #trial = make_delay_match_trial(hp, samples)
    trials.append(match_ring_task(hp, conds=1000, reps=10))
    trials.append(average_ring_task(hp, conds=2000, reps=5))

    td = TrialData(trials)

    print('Dataset created.')

    if dir_path is not None:
        data_path = os.path.join(dir_path, 'match_ring+average_ring.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(td, f)
            print(f'Dataset saved to {data_path}')

    return td



if __name__ == '__main__':
    cur_time = time.ctime().replace(' ', '_')
    dir_path = os.path.join('data', cur_time)
    hp = get_default_hp()
    mkdir_p(dir_path)
    create_dataset(hp, dir_path=dir_path)