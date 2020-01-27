
"""
make sure you're in the right conda env!

ibl docs:
https://docs.internationalbrainlab.org/en/latest/_autosummary/oneibl.one.html

"""

import pickle as pkl
import os
import itertools
import sys
import argparse

import errno

from oneibl.one import ONE
from ibllib.misc import pprint

import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.insert(1, '../')

from tools import *

eps = 1e-6

from one_tools import *


# just get the overall performance per eid for given list of eids
def get_perfs(dts):
    outs = []
    for i, eid, dt in dts:
        feedback = dt.feedback
        zeros = dt.zero_counts()

        perf = np.sum(feedback == 1) / feedback.size
        
        outs.append((i, eid, perf, zeros))

    return outs


def get_sess_data(dts):
    sdata = []
    for i, eid, dt in dts:
        try:
            dt = load_data(one, eid)
            dt.set_mask(included=True, zeros=False)
            choice = dt.choice
            contrasts = dt.contrasts
            feedback = dt.feedback

            sdata.append((i, eid, choice, contrasts, feedback))
        except KeyError as e:
            report_error(e, idx=i)

    return sdata

# helper function for smoothing filter
def create_filter(fil_type, p=None):
    if fil_type == 'flat1':
        fil_len = 21
        fil_cen = int((fil_len - 1)/2)
        fil = np.ones((fil_len))
        fil[fil_cen] = p
        for i in range(fil_cen):
            fil[fil_cen-i-1] = fil[fil_cen-i] - 0.01
            fil[fil_cen+i+1] = fil[fil_cen+i] - 0.01

    if fil_type == 'custom1':
        fil = np.asarray([1/6, 1/3, 1/3, 1/6])

    if fil_type == 'exp':
        x = (1 - p) / (1 + p) # did a bunch of math on the whiteboard to figure this out
        fil_len = 111
        fil_cen = int((fil_len - 1) / 2)
        fil = np.ones((fil_len))
        fil[fil_cen] = x
        for i in range(fil_cen):
            fil[fil_cen-1-i] = fil[fil_cen-i] * a
            fil[fil_cen+1+i] = fil[fil_cen+i] * a

    assert np.abs(np.sum(fil) - 1) < eps

    return fil

# get smoothed performance across all sessions, given a smoothing filter
def get_smoothed_perfs(dts, fil):
    fil_len = fil.size
    outs = []
    for i, eid, dt in dts:
        try:
            feedback = dt.feedback

            if feedback.size < fil_len * 2:
                continue

            feedback = (feedback + 1) / 2

            smooth = np.convolve(feedback, fil, mode='valid')
            std = np.zeros_like(smooth)

            outs.append((i, smooth, std))
            #print(f'eid {i}: {eid}')

        except KeyError as e:
            report_error(e, idx=i)

    return outs

# get average performance instead, with standard deviation
def get_averaged_perfs(dts, window=10):
    outs = []
    for i, eid, dt in dts:
        try:
            feedback = dt.feedback

            num,_ = divmod(dt.session_len, window)

            feedback = (feedback + 1) / 2

            perf_avg = []
            perf_std = []

            for j in range(num):
                fb = feedback[j*window:(j+1)*window]
                perf_avg.append(np.mean(fb))
                perf_std.append(np.var(fb))

            outs.append((i, np.asarray(perf_avg), np.asarray(perf_std)))

        except KeyError as e:
            report_error(e, idx=i)

    return outs


# compute the biases
def get_biases(dts):
    outs = []
    for i, eid, dt in dts:
        feedback = dt.feedback
        contrasts = dt.contrasts
        choice = dt.choice

        left_stimuli_count = sum(dt.contrasts > 0)
        left_correct_count = np.where(contrasts[np.where(feedback > 0)[0]] > 0)[0].size
        right_stimuli_count = sum(dt.contrasts < 0)
        right_correct_count = np.where(contrasts[np.where(feedback > 0)[0]] < 0)[0].size

        if left_stimuli_count == 0:
            left_perf = 0
        if right_stimuli_count == 0:
            right_perf = 0
        if left_stimuli_count > 0 and right_stimuli_count > 0:
            left_perf = left_correct_count / left_stimuli_count
            right_perf = right_correct_count / right_stimuli_count

        left_choice_prop = np.where(choice > 0)[0].size / dt.session_len
        left_stimuli_prop = left_stimuli_count / dt.session_len

        perf = np.sum(feedback == 1) / feedback.size
        
        outs.append((i, perf, left_perf, right_perf, left_stimuli_prop, left_choice_prop))

    return outs
