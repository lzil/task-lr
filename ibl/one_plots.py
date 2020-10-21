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
from one_compute import *

cache = 'cache'
figures = 'figures'

debug = True

    
# make simple plot for subject performance across sessions
def sess_perfs(lab, subject):

    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(fig_path)
    
    dts = load_maybe_save(one, lab, subject)
    inds, _, _, session_dts = list(zip(*dts))
    outs = get_perfs(session_dts)

    # plot all the subjects in different plots
    perfs, zeros = list(zip(*outs))
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

    if debug:
        plt.show()
    else:
        f_name = f'{fig_path}/perf.jpg'
        plt.savefig(f_name)
        print(f'Saved {f_name}')
        plt.close()


# make plots for each session of a given subject
def sess_data(lab, subject):
    fig_path = f'{figures}/{lab}/{subject}/data'
    mkdir_p(fig_path)

    dts = load_maybe_save(one, lab, subject)
    inds, _, _, session_dts = list(zip(*dts))
    outs = get_sess_data(session_dts)

    # plot all the subjects in different plots
    choices, contrasts, feedback = list(zip(*outs))

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
        sizes = 17 * np.log(16 * np.abs(contrasts[s]) + 2)

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
                s=sizes[a:b],
                edgecolors=rgba_edgecolors[a:b],
                color=rgba_colors[a:b])
            # ar.plot(np.arange(a,b),correct[a:b] * 1.4, marker='_', lw=0, ms=8, c='black')

            ar.set_ylabel('choice')
            ar.set_ylim([-1.5,1.5])
            ar.grid(b=True, which='major', axis='both')

        fig.tight_layout()
        
        if debug:
            plt.show()
            break
        else:
            f_name = f'{fig_path}/sess-{inds[s]}.jpg'
            plt.savefig(f_name)
            print(f'Saved {f_name}')
            plt.close()


# make plot for performance of a subject across all its sessions, averaged/smoothed somehow
def subject_perfs(lab, subject, p_type='average'):
    dts = load_maybe_save(one, lab, subject)
    inds, _, _, session_dts = list(zip(*dts))

    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(fig_path)

    if p_type == 'average':
        w = 40
        outs = get_averaged_perfs(session_dts, window=w)

        f_name = f'{fig_path}/perf-avg-{w}.jpg'

    elif p_type == 'smoothed':
        filter_type = 'flat1'
        p = 0.1
        fil = create_filter(filter_type, p=p)
        outs = get_smoothed_perfs(session_dts, fil=fil)

        f_name = f'{fig_path}/perf-{filter_type}-{p}.jpg'

    perfs, stds = list(zip(*outs))


    # takes a ton of code to get the right indices for the session numbers
    ivl = 500

    # sess_labels: labels in the right position, for session labels
    # sess_lens: list of running lengths, for session boundaries
    # sess_row: holds session data for each row before they're dumped into sess_labels. dumps the last row
    sess_labels = []
    sess_lens = []
    sess_row = []
    running_len = 0
    row_ivl = 0
    for i, k in enumerate(perfs):
        # get average of last running length and new running length to put session #
        running_len_old = running_len
        running_len += k.size
        avg_rl = (running_len + running_len_old) / 2
        sess_row.append((inds[i], avg_rl))
        sess_lens.append(running_len)
        # check if we have moved onto the next row
        row_ivl += k.size
        if row_ivl > ivl:
            row_ivl -= ivl
            # insert into sess_labels the right format for easy plotting with mpl
            sess_labels.append(list(zip(*sess_row)))
            sess_row = []


    sess_ind = 0
    perf_long = np.concatenate(perfs, axis=0)
    perf_std_long = np.concatenate(stds, axis=0)

    tlen = sess_lens[-1]

    nrows = divmod(tlen, ivl)[0]

    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharey=True, squeeze=False, figsize=(18,nrows*2))

    for row in range(nrows):
        ar = ax[row][0]
        a = ivl * row
        b = ivl * (row + 1)
        if b > tlen:
            b = tlen
        ar.plot(np.arange(a,b), perf_long[a:b], lw=1.5)
        ar.fill_between(np.arange(a,b), perf_long[a:b], perf_long[a:b] + perf_std_long[a:b], color='salmon', alpha='0.3')
        ar.fill_between(np.arange(a,b), perf_long[a:b], perf_long[a:b] - perf_std_long[a:b], color='salmon', alpha='0.3')


        ar.set_xlim([a, b])
        ar.set_ylim([0, 1])
        ar.vlines(sess_lens, ymin=0, ymax=1, linestyles='solid', linewidths=0.5)
        ar.set_xticks(sess_labels[row][1])
        ar.set_xticklabels(sess_labels[row][0])
        ar.set_ylabel('performance')
        ar.grid(b=True, which='major', axis='y')
        ar.set_yticks(np.arange(0, 1.1, step=0.5))

    plt.xlabel('session')

    fig.tight_layout()

    plt.gcf()

    if debug:
        plt.show()
    else:
        plt.savefig(f_name)
        print(f'Saved {f_name}')
        plt.close()


# make plot with performances for the top 6 subjects of a lab
def all_perfs(lab, cache='cache', figures='figures'):
    fig_path = f'{figures}/{lab}'
    mkdir_p(fig_path)

    eids, sinfos = one.search(lab=lab, details=True)
    subjects = get_subject_counts(sinfos)
    subjects, counts = zip(*subjects[:9])

    fig, ax = plt.subplots(nrows=3, ncols=3, sharey=True, squeeze=False, figsize=(14, 9))

    for ind, subject in enumerate(subjects):
        print(f'Starting on subject: {subject}')

        dts = load_maybe_save(one, lab, subject)
        inds, _, _, session_dts = list(zip(*dts))
        outs = get_perfs(session_dts)

        perfs, _ = list(zip(*outs))
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

    if debug:
        plt.show()
    else:
        f_name = f'{fig_path}/perf.jpg'
        plt.savefig(f_name)
        print(f'Saved {f_name}')
        plt.close()
    


# make plot for biases that a subject might have
def subject_biases(lab, subject):
    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(fig_path)

    dts = load_maybe_save(one, lab, subject)
    inds, _, _, session_dts = list(zip(*dts))
    outs = get_biases(session_dts)

    perfs, lperfs, rperfs, lsprop, lcprop = list(zip(*outs))
    plt.figure(figsize=(18, 5))
    plt.plot(inds, lperfs, '-', c='salmon', lw=1, label='left stimulus performance')
    plt.plot(inds, rperfs, '-', c='royalblue', lw=1, label='right stimulus performance')
    plt.plot(inds, lsprop, '-', c='darkorange', lw=2, label='left stimulus proportion')
    plt.plot(inds, lcprop, '-', c='darkgreen', lw=2, label='left choice proportion')

    plt.grid(b=True, which='major', axis='both')
    plt.title(subject)
    plt.xlabel('session')
    plt.ylabel('performance / proportion')
    plt.axis([0,None,0,1])
    plt.fill_between(inds, perfs, lperfs, color='salmon', alpha=0.3)
    plt.fill_between(inds, perfs, rperfs, color='royalblue', alpha=0.3)
    plt.yticks(np.arange(0, 1, step=0.1))

    plt.plot(inds, perfs, 'o-', c='darkorchid', lw=2, label='overall performance')

    plt.legend()

    plt.tight_layout()

    if debug:
        plt.show()
    else:
        f_name = f'{fig_path}/bias.jpg'
        plt.savefig(f_name)
        print(f'Saved {f_name}')
        plt.close()



def wheel_positions(lab, subject, session_num):
    fig_path = f'{figures}/{lab}/{subject}'
    mkdir_p(fig_path)

    eids, sinfos = one.search(lab=lab, details=True, subject=subject)
    eids, sinfos = order_eids(eids, sinfos)

    eid = eids[session_num]

    d_types = [
        'trials.choice',
        'trials.contrastLeft',
        'trials.contrastRight',
        'trials.feedbackType',
        'trials.choice',
        'trials.included',
        'trials.response_times',
        'trials.goCue_times',
        'trials.intervals',
        'wheel.position',
        'wheel.timestamps'
    ]

    sess_data = load_data(one, eid, d_types=d_types)

    groups = sess_data.wheel_indices()
    wt = sess_data.wt
    wp = sess_data.wp

    start = 50
    stop = 70
    running_break = 0
    last_pos = 0
    plt.figure(figsize=(18, 2))
    for i in range(start, stop):
        if len(groups[i][1]) == 0:
            print(f'skipping {i}, nothing in the group')
            continue
        timestamps = wt[groups[i][1]]
        if last_pos != 0:
            running_break += wt[groups[i][1][0]] - last_pos
            timestamps -= running_break
        last_pos = wt[groups[i][1][-1]]
        positions = wp[groups[i][1]] - wp[groups[i][1][0]]
        plt.plot(timestamps, positions, '-', lw=1, c='dodgerblue')

        cxc = sess_data._cxc[i]
        if cxc > 0:
            color = 'lightgreen'
        else:
            color = 'lightcoral'

        plt.axvspan(timestamps[0], timestamps[-1], facecolor=color, alpha=abs(cxc))
        plt.vlines(timestamps[0], ymin=-200, ymax=200, linestyle='solid', linewidth=1,color='black')

    plt.ylim([-.5,.5])
    plt.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False)
    plt.xlim([wt[groups[start][1][0]]-.1, timestamps[-1]+.1])


    if debug:
        plt.show()
    else:
        f_name = f'{fig_path}/wheel-{session}-{start}-{stop}.jpg'
        plt.savefig(f_name)
        print(f'Saved {f_name}')
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = [
        'sess_data',
        'subject_perfs',
        'all_perfs',
        'sess_perfs',
        'subject_bias',
        'wheel_positions'
    ]
    parser.add_argument('action', choices=actions)
    parser.add_argument('-l', '--lab')
    parser.add_argument('-s', '--subject')
    parser.add_argument('-n', '--session', type=int)

    args = parser.parse_args()

    assert args.lab is not None

    global one
    one = ONE()

    if args.action == 'sess_data':
        assert args.subject is not None
        sess_data(args.lab, args.subject)

    elif args.action == 'subject_perfs':
        assert args.subject is not None
        subject_perfs(args.lab, args.subject)

    elif args.action == 'all_perfs':
        all_perfs(args.lab)

    elif args.action == 'sess_perfs':
        assert args.subject is not None
        sess_perfs(args.lab, args.subject)

    elif args.action == 'subject_bias':
        assert args.subject is not None
        subject_biases(args.lab, args.subject)

    elif args.action == 'wheel_positions':
        assert args.subject is not None
        assert args.session is not None
        wheel_positions(args.lab, args.subject, args.session)




