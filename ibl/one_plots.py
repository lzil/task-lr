"""
make sure you're in the right conda env!

ibl docs:
https://docs.internationalbrainlab.org/en/latest/_autosummary/oneibl.one.html

"""

import pickle as pkl
import os
import itertools
import sys

import errno

from oneibl.one import ONE
from ibllib.misc import pprint

import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.insert(1, '../')

from tools import *


def sess_perfs(lab, subject, cache='cache', figures='figures'):
    one = ONE()

    cache_path = f'{cache}/{lab}/{subject}'
    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(cache_path)
    mkdir_p(fig_path)

    eids, sinfos = one.search(lab=lab, details=True, subject=subject)
    eids, sinfos = order_eids(eids, sinfos)
    
    # if performances are already done no point recomputing
    f_name = f'{cache_path}/perf.pkl'
    if not os.path.isfile(f_name):
        print("Stored file doesn't exist; downloading and calculating.")
        outs = get_perfs(eids)
        with open(f_name, 'wb') as f:
            pkl.dump(outs, f)
    else:
        print("Stored file exists; loading and moving on.")
        with open(f_name, 'rb') as f:
            outs = pkl.load(f)

    # plot all the subjects in different plots
    inds, eids, perfs, zeros = list(zip(*outs))
    plt.figure(figsize=(18, 5))
    plt.plot(inds, perfs, 'ro-')
    for j in range(len(outs)):
        plt.annotate(zeros[j], (inds[j], perfs[j]), fontsize='small')
    plt.grid(b=True, which='major', axis='x')
    plt.title(subject)
    plt.xlabel('session')
    plt.ylabel('avg performance')
    plt.axis([None,None,0,1])
    plt.tight_layout()
    plt.savefig(f'{fig_path}/perf.jpg')
    plt.close()


def sess_data(lab, subject, cache='cache', figures='figures'):
    cache_path = f'{cache}/{lab}/{subject}'
    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(fig_path)
    mkdir_p(cache_path)

    eids, sinfos = one.search(lab=lab, details=True, subject=subject)
    eids, sinfos = order_eids(eids, sinfos)
    
    # if performances are already done no point recomputing
    f_name = f'{cache_path}/data.pkl'
    print(f"Attempting to load {f_name}...")
    if not os.path.isfile(f_name):
        print("Stored file doesn't exist; loading/downloading and calculating.")
        outs = sess_data(eids)
        with open(f_name, 'wb') as f:
            pkl.dump(outs, f)
    else:
        print("Stored file exists; loading and moving on.")
        with open(f_name, 'rb') as f:
            outs = pkl.load(f)

    # plot all the subjects in different plots
    inds, eids, choices, contrasts, feedback = list(zip(*outs))

    # iterate through all sessions
    for s in range(len(inds)):
        tlen = contrasts[s].size
        cxc = choices[s] * contrasts[s]

        correct = np.sign(contrasts[s])
        rgba_colors = np.zeros((tlen, 4))
        rgba_colors[:, 1] = cxc > 0
        rgba_colors[:, 0] = cxc < 0
        rgba_colors[:, 3] = np.abs(contrasts[s])
        rgba_edgecolors = rgba_colors[:, :3]

        nrows, extra = divmod(tlen, 150)
        if extra > 0:
            nrows += 1
        fig, ax = plt.subplots(nrows=nrows, ncols=1, sharey=True, squeeze=False, figsize=(18, 1.2*nrows))
        for row in range(nrows):
            ar = ax[row][0]
            a = 150 * row
            b = 150 * (row + 1)
            if b > tlen:
                b = tlen
            ar.scatter(
                x=np.arange(a,b),
                y=choices[s][a:b],
                marker='o',
                s=49,
                edgecolors=rgba_edgecolors[a:b],
                color=rgba_colors[a:b])
            ar.plot(np.arange(a,b),correct[a:b] * 1.4, marker='_', lw=0, ms=8, c='black')

            ar.set_ylabel('choice')
            ar.set_ylim([-1.5,1.5])
            ar.grid(b=True, which='major', axis='both')

        fig.tight_layout()
        
        f_name = f'{fig_path}/{inds[s]}-data.jpg'
        plt.savefig(f_name)
        print(f"Processed session {inds[s]}")
        plt.close()



def subject_smoothed_perfs(lab, cache='cache', figures='figures'):
    filter_type = 'flat1'
    p = 0.1
    
    fil = create_filter(filter_type, p=p)

    eids, sinfos = one.search(lab=lab, details=True, subject=subject)
    eids, sinfos = order_eids(eids, sinfos)

    cache_path = f'{cache}/{lab}/{subject}'
    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(fig_path)
    mkdir_p(cache_path)


    # if performances are already done no point recomputing
    f_name = f'{cache_path}/{filter_type}-{p}.pkl'
    if not os.path.isfile(f_name):
        print("Stored file doesn't exist; downloading and calculating.")
        outs = get_smoothed_perfs(eids, fil=fil)
        with open(f_name, 'wb') as f:
            pkl.dump(outs, f)
    else:
        print("Stored file exists; loading and moving on.")
        with open(f_name, 'rb') as f:
            outs = pkl.load(f)

    inds, eids, perfs = list(zip(*outs))
    sess_lens = []

    running_len = 0
    for k in perfs:
        running_len += k.size
        sess_lens.append(running_len)

    sess_ind = 0
    perf_long = np.concatenate(perfs, axis=0)

    tlen = sess_lens[-1]

    nrows = divmod(tlen, 10000)[0]

    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharey=True, squeeze=False, figsize=(18,nrows*2))

    for row in range(nrows):
        ar = ax[row][0]
        a = 10000 * row
        b = 10000 * (row + 1)
        if b > tlen:
            b = tlen
        ar.plot(np.arange(a,b), perf_long[a:b], lw=0.5)
        ar.set_xlim([a, b])
        ar.set_ylim([0, 1])
        ar.vlines(sess_lens, ymin=0, ymax=1, linestyles='dashed', linewidths=0.5)
        ar.set_ylabel('performance')
        #ar.set_xlim([0,None])

    # plot all the subjects in different plots
    # inds, eids, perfs = list(zip(*outs))
    fig.tight_layout()
    plt.savefig(f'{fig_path}/{filter_type}-{p}.jpg')
    plt.close()



def all_perfs(lab, cache='cache', figures='figures'):
    eids, sinfos = one.search(lab=lab, details=True)
    subjects = get_subject_counts(sinfos)
    subjects, counts = zip(*subjects[:6])

    fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, squeeze=False, figsize=(14, 6))

    for ind, subject in enumerate(subjects):
        print(f'Starting on subject: {subject}')
        cache_path = f'{cache}/{lab}/{subject}'
        mkdir_p(cache_path)
        
        eids, sinfos = one.search(lab=lab, details=True, subject=subject)
        eids, sinfos = order_eids(eids, sinfos)
    
        # if performances are already done no point recomputing
        f_name = f'{cache_path}/perf.pkl'
        if not os.path.isfile(f_name):
            print("Stored file doesn't exist; downloading and calculating.")
            outs = get_perfs(eids)
            with open(f_name, 'wb') as f:
                pkl.dump(outs, f)
        else:
            print("Stored file exists; loading and moving on to the next subject.")
            with open(f_name, 'rb') as f:
                outs = pkl.load(f)

        inds, eids, perfs, _ = list(zip(*outs))
        row, col = divmod(ind, 3)
        arc = ax[row,col]
        arc.plot(inds, perfs, 'r.-')
        arc.set_ylim([0, 1])
        arc.set_title(subject)
        arc.grid(b=True, which='major', axis='x')
        arc.set_xlabel('session')
        arc.set_ylabel('perf')
    
    plt.gcf()
    plt.tight_layout()
    fig_path = f'{figures}/{lab}'
    mkdir_p(fig_path)
    plt.savefig(f'{fig_path}/perf.jpg')
    plt.show()