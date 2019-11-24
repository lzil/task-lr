import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

task_map = {
    'do_nothing': 0,
    'delay_match': 1
}

def default_for(var, val):
    if var is not None:
        return var
    return val


class Trial:
    def __init__(self, hp, n_trials=100):
        self.n_trials = n_trials
        self.n_steps = hp['n_steps']
        self.n_ring = hp['n_ring']
        self.n_tasks = hp['n_tasks']
        self.n_feats = hp['n_features']
        self.n_output = hp['n_output']

        self.n_stim_steps = int(hp['n_steps'] * hp['stim_frac'])

        # fixation stimulus is always 1 for stimulus steps, then 0
        self.x_fix = np.concatenate([
                np.ones([self.n_trials, self.n_stim_steps, 1]),
                np.zeros([self.n_trials, self.n_steps - self.n_stim_steps, 1])
                ], axis=1
            )

        self.y_fix = np.concatenate([
                np.ones([self.n_trials, self.n_stim_steps, 1]),
                np.zeros([self.n_trials, self.n_steps - self.n_stim_steps, 1])
                ], axis=1
            )

        # identify task through task id later
        self.x_task = np.zeros([self.n_trials, self.n_steps, self.n_tasks])

        # placeholder
        self.x_ring = np.zeros([self.n_trials, self.n_steps, self.n_ring])

        # placeholder
        self.x_feats = np.zeros([self.n_trials, self.n_steps, self.n_feats])

        # placeholder
        self.y_resp = np.zeros([self.n_trials, self.n_steps, self.n_output])



    # replace input values for ring and features
    def put(self, task_id, ix=[0,None,0,None], ring=None, feats=None, resp=None):
        # indices 0 and 1 are batch start and stop points
        # indices 2 and 3 are step start and stop points
        if ix[0] is None:
            ix[0] = 0
        if ix[1] is None:
            ix[1] = self.n_trials
        if ix[2] is None:
            ix[2] = 0
        if ix[3] is None:
            ix[3] = self.n_steps

        # set task id
        self.x_task[ix[0]:ix[1],task_id] = 1
        # randomly picking number of each type of permutations
        if ring is not None:
            self.x_ring[ix[0]:ix[1],ix[2]:ix[3],:] = ring
        if feats is not None:
            self.x_feats[ix[0]:ix[1],ix[2]:ix[3],:] = feats
        if resp is not None:
            self.y_resp[ix[0]:ix[1],ix[2]:ix[3],:] = resp


    def get_trial_data(self):
        return [self.x_fix, self.x_task, self.x_ring, self.x_feats, self.y_fix, self.y_resp]



class TrialData(Dataset):
    def __init__(self, trials):
        xy = [[] for i in range(6)]
        for trial in trials:
            trial_xy = trial.get_trial_data()
            for i, z in enumerate(trial_xy):
                xy[i].append(z)


        # concatenate along n_trials dimension
        self.xy = [torch.from_numpy(
            np.concatenate(i)
            ).to(torch.float) for i in xy]

        
    def __len__(self):
        return self.xy[0].size()[0]

    def __getitem__(self, idx):
        sample = {
            'x_fix': self.xy[0][idx],
            'x_task': self.xy[1][idx],
            'x_ring': self.xy[2][idx],
            'x_feats': self.xy[3][idx],
            'y_fix': self.xy[4][idx],
            'y_resp': self.xy[5][idx]
            }
            
        return sample