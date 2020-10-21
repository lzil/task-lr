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
from one_tools import *




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





if __name__ == '__main__':
    one = ONE()

    cache = 'cache'
    figures = 'figures'

    lab = 'hoferlab'
    subject = 'SWC_015'

    eids, sinfos = one.search(lab=lab, details=True, subject=subject)
    eids, sinfos = order_eids(eids, sinfos)

    outs = []

    cache_path = f'{cache}/{lab}/{subject}'
    f_name = f'{cache_path}/raw.pkl'

    sesses = load_maybe_save(one, lab, subject)

    print(sesses[25], sesses[75])

    


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
