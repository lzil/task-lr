import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from tools import *


task_id_map = {
    'do_nothing':0,
    'match_feats':1,
    'match_ring':2,
    'average_ring':3
}

class Trial:
    def __init__(self, hp, n_trials=100):
        self.n_trials = n_trials
        self.n_steps = hp['n_steps']
        self.n_tasks = hp['n_tasks']
        self.n_in_feats = hp['n_in_features']
        self.n_in_ring = hp['n_in_ring']
        self.n_out_choice = hp['n_out_choice']
        self.n_out_ring = hp['n_out_ring']

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
        self.x_ring = np.zeros([self.n_trials, self.n_steps, self.n_in_ring])

        # placeholder
        self.x_feats = np.zeros([self.n_trials, self.n_steps, self.n_in_feats])

        # placeholder
        self.y_choice = np.zeros([self.n_trials, self.n_steps, self.n_out_choice])

        # placeholder
        self.y_ring = np.zeros([self.n_trials, self.n_steps, self.n_out_ring])

    # helper function to replace None values in indices list with default values
    def _get_indices(self, ix):
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
        return ix

    # replace input/output values for ring and features
    def put(self, section, mode, ix=[None,None,None,None], vals=None):
        if section == 'task':
            arr = self.x_task
        elif section == 'in_ring':
            arr = self.x_ring
        elif section == 'in_feats':
            arr = self.x_feats
        elif section == 'out_choice':
            arr = self.y_choice
        elif section == 'out_ring':
            arr = self.y_ring

        ix = self._get_indices(ix)
        arr_ind = arr[ix[0]:ix[1],ix[2]:ix[3],:]

        if mode == 'vals':
            shp = arr_ind.shape
            arr[ix[0]:ix[1],ix[2]:ix[3],:] = np.tile(np.copy(vals),(shp[0],shp[1],1))
        elif mode == 'ring_loc':
            ang, gamma, var = vals
            n_units = self.n_in_ring if section == 'in_ring' else self.n_out_ring
            for u in range(n_units):
                u_dist = np.abs(ang - angle(n_units, u)) % (2 * np.pi)
                u_min_dist = np.minimum(u_dist, 2*np.pi - u_dist)
                u_val = gamma * 0.8 * np.exp(-1/2 * np.square(8/np.pi * u_min_dist) / var)
                arr_ind[:,:,u] += u_val

    def get_trial_data(self):
        return [self.x_fix, self.x_task, self.x_ring, self.x_feats, self.y_fix, self.y_choice, self.y_ring]



class TrialData(Dataset):
    def __init__(self, trials):
        xy = [[] for i in range(7)]
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
            'y_choice': self.xy[5][idx],
            'y_ring': self.xy[6][idx]
            }
            
        return sample



# test for delay match task
def make_delay_match_trial(hp, n_trials):

    task = Trial(hp, n_trials=n_trials)

    n_in_feats = hp['n_in_features']
    n_out_choice = hp['n_out_choice']
    rn = np.random.choice(range(1, n_trials), size=n_in_feats-1, replace=False)
    rn = np.concatenate((np.array([0]), np.sort(rn), np.array([n_trials])))

    task_vals = one_hot(hp['n_tasks'], hp['task_id'])
    task.put('task', 'vals', vals=task_vals)
    for i in range(n_in_feats):
        # set the stimulus and response values
        v_feat = one_hot(n_in_feats, i)
        v_choice = one_hot(n_out_choice, i)
        # features stimulus and response values
        task.put('in_feats', 'vals', ix=[rn[i],rn[i+1],None,task.n_stim_steps], vals=v_feat)
        task.put('out_choice', 'vals', ix=[rn[i],rn[i+1],task.n_stim_steps,None], vals=v_choice)

    return task


def match_ring_task(hp, n_trials):
    task = Trial(hp, n_trials=n_trials)

    angles_loc = np.random.uniform(0, 2*np.pi, size=(n_trials))
    task_vals = one_hot(hp['n_tasks'], task_id_map['match_ring'])
    task.put('task', 'vals', vals=task_vals)
    for i in range(n_trials):
        ang = angles_loc[i]
        # features stimulus and response values
        task.put('in_ring', 'ring_loc', ix=[i,i+1,None,task.n_stim_steps], vals=[ang, 1, 1])
        task.put('out_ring', 'ring_loc', ix=[i,i+1,task.n_stim_steps,None], vals=[ang, 1, 1])

    return task


def average_ring_task(hp, n_trials):
    task = Trial(hp, n_trials=n_trials)

    angles_loc = np.random.uniform(0, 2*np.pi, size=(2, n_trials))
    task_vals = one_hot(hp['n_tasks'], task_id_map['average_ring'])
    task.put('task', 'vals', vals=task_vals)
    for i in range(n_trials):
        ang1 = angles_loc[0,i]
        ang2 = angles_loc[1,i]
        avg_ang = (ang1 + ang2) / 2
        # features stimulus and response values
        task.put('in_ring', 'ring_loc', ix=[i,i+1,None,task.n_stim_steps], vals=[ang1, 1, 1])
        task.put('in_ring', 'ring_loc', ix=[i,i+1,None,task.n_stim_steps], vals=[ang2, 1, 1])
        task.put('out_ring', 'ring_loc', ix=[i,i+1,task.n_stim_steps,None], vals=[avg_ang, 1, 1])

    return task
