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
import one_plots




# def all_seq_stats(eids, sinfos):
#     for i,eid in enumerate(eids):
#         data = load_data(eid)
#         tmp = get_session_feedback(data, idx=i)
#         if type(tmp) == TypeError:
#             continue
#         cxcs, feedback = tmp

#         perfs = []

#         counts = np.zeros(15)
#         cur_combo = 0
#         cur_val = -1
#         for j in range(feedback_cut.size):
#             if feedback_cut[j] == cur_val:
#                 cur_combo += 1
#             else:
#                 cur_val = feedback_cut[j]
#                 if cur_combo > counts.size:
#                     print(f'Warning, combo size {cur_combo} but only {counts.size} elements in counts.')
#                     cur_combo = counts.size
#                 counts[cur_combo - 1] += 1
#                 cur_combo = 1


#         outs.append((i, eid, counts))

#     return outs


# def calc_seq_stats(eids, sinfos):
#     outs = []
#     good_eids = 0
#     for i,eid in enumerate(eids):

#         data = load_data(tmp = get_session_, idx=ifeedback(data)
#             if type(tmp) == TypeError:
#                 continue
#             cxcs, feedback = tmp

#         ) / 2

#         if feedback_cut.size < 250:
#             continue


#         perfs = []

#         counts = np.zeros(15)
#         cur_combo = 0
#         cur_val = -1
#         for j in range(feedback_cut.size):
#             if feedback_cut[j] == cur_val:
#                 cur_combo += 1
#             else:
#                 cur_val = feedback_cut[j]
#                 if cur_combo > counts.size:
#                     print(f'Warning, combo size {cur_combo} but only {counts.size} elements in counts.')
#                     cur_combo = counts.size
#                 counts[cur_combo - 1] += 1
#                 cur_combo = 1


#         #perf = np.sum(feedback_cut == 1) / feedback_cut.size
#         outs.append((i, eid, counts))
#         #print(f'eid {i}: {eid}, performance: {perf}')

#         good_eids += 1

#         if good_eids >= 25:
#             break

#     return outs


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


if __name__ == '__main__':
    one = ONE()

    cache = 'cache'
    figures = 'figures'



    setting = "sess_data"

    lab = 'hoferlab'
    subject = 'SWC_015'


    if setting == 'sess_data':
        one_plots.sess_data(lab, subject, cache, figures)

    if setting == 'subject_smoothed_perfs':
        one_plots.subject_smoothed_perfs(lab, subject, cache, figures)

    if setting == 'all_perfs':
        one_plots.all_perfs(lab, cache, figures)

    if setting == 'sess_perfs':
        one_plots.sess_perfs(lab, subject, cache, figures)
        
    # elif setting == 'seq_stats':

    #     f_name = c_prefix + '-stats.pkl'
    #     if not os.path.isfile(f_name):

    #         outs = calc_seq_stats(eids, sinfos)

    #         # save the performances
    #         with open(f_name, 'wb') as f:
    #             pkl.dump(outs, f)

    #     else:

    #         with open(f_name, 'rb') as f:
    #             outs = pkl.load(f)


    #     rows = 5
    #     cols = 5
    #     fig, ax = plt.subplots(nrows=rows, ncols=cols, sharey=True, squeeze=True)
    #     for r in range(rows):
    #         for c in range(cols):
    #             ax[r, c].plot(outs[rows*r+c][2])
    #             ax[r, c].set_title(outs[rows*r+c][0])
    #             ax[r,c].set_xticks(np.arange(16))
    #             ax[r,c].grid(b=True, which='major', color='r', alpha=0.5)

    #     plt.gcf()
    #     plt.show()
