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

sys.path.insert(1, '../../')


from network import *
from trial import *

from tools import *

from train import *


# generate data loader
def get_data_loader(hp, data_path=None):

    if data_path is None:
        sys.exit(0)
    else:
        print('Loading dataset...')
        with open(data_path, 'rb') as f:
            td = pickle.load(f)

        dl = DataLoader(
            dataset=td,
            batch_size=hp['batch_size'],
            shuffle=True,
            drop_last=True
            )
        print('Dataset loaded.')

    return dl


def create_dataset(hp, dir_path):

    samples = hp['n_samples']

    print('Creating dataset...')

    # list the trials you want to do here
    #trial = make_delay_match_trial(hp, samples)
    #trial = match_ring_task(hp, samples)
    trial = average_ring_task(hp, samples)

    td = TrialData([trial])

    print('Dataset created.')

    with open(os.path.join(dir_path, '1'), 'wb') as f:
        
        pickle.dump(td, f)



if __name__ == '__main__':
    w_path = os.path.join('data', time.ctime().replace(' ', '_'))
    hp = get_default_hp()
    mkdir_p(w_path)
    create_dataset(hp, dir_path=w_path)