import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import pdb

from torch.utils.data import Dataset, DataLoader

from tools import *


task_id_map = {
    'do_nothing':0,
    'match_feats':1,
    'match_ring':2,
    'average_ring':3
}

trial_hp = {
    'fix_start_range': [0, 0.3],
    'fix_lens_range': [0.2, 0.5],
    'ang_var': 0.5,
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

        self.sigma_x = hp['sigma_x']

        self.x = {
            'fix': np.zeros([self.n_steps, 1]),
            'task': np.zeros([self.n_steps, self.n_tasks]),
            'feats': np.zeros([self.n_steps, self.n_in_feats]),
            'ring': np.zeros([self.n_steps, self.n_in_ring])
        }

        self.y = {
            'fix': np.zeros([self.n_steps, 1]),
            'choice': np.zeros([self.n_steps, self.n_out_choice]),
            'ring': np.zeros([self.n_steps, self.n_out_ring])
        }
    

    # helper function to replace None values in indices list with default values
    def _get_indices(self, ix):
        # indices 0 and 1 are trials start and stop points
        # indices 2 and 3 are step start and stop points
        if ix[0] is None:
            ix[0] = 0
        if ix[1] is None:
            ix[1] = self.n_steps
        return ix

    # replace values
    def put(self, section, stype, mode, ix=[None,None], vals=None):
        if section == 'x':
            arr = self.x[stype]
        elif section == 'y':
            arr = self.y[stype]

        if stype == 'fix':
            arr[:,0] = np.copy(vals)
        else:
            ix = self._get_indices(ix)
            val_resized = np.tile(np.copy(vals), (ix[1] - ix[0], 1))

            # val_resized should be (n_steps, *) shape
            if mode == 'write':
                arr[ix[0]:ix[1],:] = val_resized
            elif mode == 'append':
                # should only be used on inputs for now; i.e. section == 'x'
                arr[ix[0]:ix[1],:] += val_resized


    # create a vector that stores gaussian-distributed values for use in rings
    def get_ring_vec(self, n_units, ang, gamma, var):
        v = np.empty((n_units))
        for u in range(n_units):
            u_dist = np.abs(ang - angle(n_units, u)) % (2 * np.pi)
            u_min_dist = np.minimum(u_dist, 2*np.pi - u_dist)
            u_val = gamma * 0.8 * np.exp(-1/2 * np.square(8/np.pi * u_min_dist) / var)
            v[u] += u_val
        return v


    # add some noise to the input
    def x_noise(self, x):
        return np.random.randn(*x.shape) * self.sigma_x

    # add noise and return the dataset
    def get_trial_data(self):
        single_data = [
            self.x['fix'],
            self.x['task'],
            self.x['feats'],
            self.x['ring'],
            self.y['fix'],
            self.y['choice'],
            self.y['ring'],
        ]
        trial_data = [np.tile(z, (self.n_trials, 1, 1)) for z in single_data]
        for x in trial_data[:4]:
            x += self.x_noise(x)
        return trial_data


class Trial2:
    def __init__(self, hp, n_trials=100):
        self.n_trials = n_trials
        self.n_steps = hp['n_steps']
        self.n_tasks = hp['n_tasks']
        self.n_in_feats = hp['n_in_features']
        self.n_in_ring = hp['n_in_ring']
        self.n_out_choice = hp['n_out_choice']
        self.n_out_ring = hp['n_out_ring']

        self.sigma_x = hp['sigma_x']
        self.stim_sigma = hp['stim_sigma']

        self.n_stim_steps = int(hp['n_steps'] * hp['fix_frac'])

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
        # indices 0 and 1 are trials start and stop points
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
        elif section == 'in_fix':
            arr = self.x_fix
        elif section == 'in_ring':
            arr = self.x_ring
        elif section == 'in_feats':
            arr = self.x_feats
        elif section == 'out_fix':
            arr = self.y_fix
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

    # add some noise to the input
    def add_x_noise(self):
        x_list = [self.x_fix, self.x_task, self.x_ring, self.x_feats]
        for x in x_list:
            x += np.random.randn(x.shape) * self.sigma_x

    def get_trial_data(self):
        self.add_x_noise()
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
            'x_feats': self.xy[2][idx],
            'x_ring': self.xy[3][idx],
            'y_fix': self.xy[4][idx],
            'y_choice': self.xy[5][idx],
            'y_ring': self.xy[6][idx],
            }
            
        return sample


### HELPER FUNCTIONS


def get_fix_start_ends(hp, conds):
    flr = trial_hp['fix_lens_range']
    fsr = trial_hp['fix_start_range']
    fix_lens = (hp['n_steps'] * np.random.uniform(flr[0],flr[1],size=(conds))).astype(np.int)
    fix_start = (hp['n_steps'] * np.random.uniform(fsr[0],fsr[1],size=(conds))).astype(np.int)
    fix_end = fix_start + fix_lens

    return fix_start, fix_end



### TASKS


# # test for delay match task
# def delay_match_trial(hp, n_trials):

#     task = Trial(hp, n_trials=n_trials)

#     n_in_feats = hp['n_in_features']
#     n_out_choice = hp['n_out_choice']
#     rn = np.random.choice(range(1, n_trials), size=n_in_feats-1, replace=False)
#     rn = np.concatenate((np.array([0]), np.sort(rn), np.array([n_trials])))

#     task_vals = one_hot(hp['n_tasks'], hp['task_id'])
#     task.put('task', 'vals', vals=task_vals)
#     for i in range(n_in_feats):
#         # set the stimulus and response values
#         v_feat = one_hot(n_in_feats, i)
#         v_choice = one_hot(n_out_choice, i)
#         # features stimulus and response values
#         task.put('in_feats', 'vals', ix=[rn[i],rn[i+1],None,task.n_stim_steps], vals=v_feat)
#         task.put('out_choice', 'vals', ix=[rn[i],rn[i+1],task.n_stim_steps,None], vals=v_choice)

#     return task


def match_ring_task(hp, conds, reps):
    """Match the orientation on the input ring 

    X:
    [fix_start, fix_end] Fixation
    [fix_start, fix_end] Ring orientation

    Y:
    [fix_start, fix_end] Fixation
    [fix_end, END] Ring orientation (same as input)


    """
    total_trials = conds * reps

    fix_start, fix_end = get_fix_start_ends(hp, conds)
    angles_loc = np.random.uniform(0, 2*np.pi, size=(conds))

    trials = []

    for c in range(conds):
        task = Trial(hp, n_trials=reps)
        
        task_vals = one_hot(hp['n_tasks'], task_id_map['match_ring'])
        task.put('x', 'task', 'write', ix=[None,None], vals=task_vals)

        ang = angles_loc[c]

        v_fix = np.zeros((hp['n_steps']))
        v_fix[fix_start[c]:fix_end[c]] = 1
        task.put('x', 'fix', 'write', ix=[None,None], vals=v_fix)
        task.put('y', 'fix', 'write', ix=[None,None], vals=v_fix)
        # features stimulus and response values
        x_ring = task.get_ring_vec(hp['n_in_ring'], ang, 1, trial_hp['ang_var'])
        y_id = np.round(ang / (2 * np.pi) * hp['n_out_ring']).astype(np.int) % hp['n_out_ring']
        y_ring = one_hot(hp['n_out_ring'], y_id)
        task.put('x', 'ring', 'write', ix=[fix_start[c],fix_end[c]], vals=x_ring)
        task.put('y', 'ring', 'write', ix=[fix_end[c],None], vals=y_ring)

        trials.append(task)

    return trials


def average_ring_task(hp, conds, reps):
    """Average the two orientations on the input rings

    X:
    [fix_start, fix_end] Fixation
    [fix_start, fix_end] Ring orientation 1
    [fix_start, fix_end] Ring orientation 2

    Y:
    [fix_start, fix_end] Fixation
    [fix_end, END] Ring orientation (at average location of input rings)


    """
    total_trials = conds * reps

    fix_start, fix_end = get_fix_start_ends(hp, conds)
    angles_loc = np.random.uniform(0, 2*np.pi, size=(2, conds))

    trials = []

    for c in range(conds):
        task = Trial(hp, n_trials=reps)
        
        task_vals = one_hot(hp['n_tasks'], task_id_map['average_ring'])
        task.put('x', 'task', 'write', ix=[None,None], vals=task_vals)

        v_fix = np.zeros((hp['n_steps']))
        v_fix[fix_start[c]:fix_end[c]] = 1
        task.put('x', 'fix', 'write', ix=[None,None], vals=v_fix)
        task.put('y', 'fix', 'write', ix=[None,None], vals=v_fix)

        ang1 = angles_loc[0,c]
        ang2 = angles_loc[1,c]
        avg_ang = (ang1 + ang2) / 2
        if abs(ang1 - ang2) > np.pi:
            avg_ang = (avg_ang + np.pi) % (2 % np.pi)

        # features stimulus and response values
        x_ring1 = task.get_ring_vec(hp['n_in_ring'], ang1, 1, trial_hp['ang_var'])
        x_ring2 = task.get_ring_vec(hp['n_in_ring'], ang2, 1, trial_hp['ang_var'])
        y_id = np.round(avg_ang / (2 * np.pi) * hp['n_out_ring']).astype(np.int) % hp['n_out_ring']
        y_ring = one_hot(hp['n_out_ring'], y_id)
        task.put('x', 'ring', 'write', ix=[fix_start[c],fix_end[c]], vals=x_ring1)
        task.put('x', 'ring', 'append', ix=[fix_start[c],fix_end[c]], vals=x_ring2)
        task.put('y', 'ring', 'write', ix=[fix_end[c],None], vals=y_ring)

        trials.append(task)

    return trials


