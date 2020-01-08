"""
make sure you're in the right conda env!

ibl docs:
https://docs.internationalbrainlab.org/en/latest/_autosummary/oneibl.one.html

"""

import pickle as pkl
import os
import itertools

from oneibl.one import ONE
from ibllib.misc import pprint

import numpy as np
import matplotlib.pyplot as plt

import pdb

eps = 1e-6


DEFAULT_D_TYPES = [
    'trials.choice',
    'trials.contrastLeft',
    'trials.contrastRight',
    'trials.feedbackType',
    'trials.choice',
    'trials.included'
]

class SessionData():
    def __init__(self, data):
        self.choice = data['trials.choice']
        self.contrast_left = data['trials.contrastLeft']
        self.contrast_right = data['trials.contrastRight']
        self.feedback = data['trials.feedbackType']
        self.choice = data['trials.choice']
        self.included = data['trials.included']


def generate_acc(size, acc):
    ...


# return counts for each of the subjects
# input: a list of details dictionaries (found via one.search with details=True)
# output: a list of tuples of the form (subject, #) in descending order of #
def get_subject_counts(sinfos):
    subjs = {}
    for info in sinfos:
        if info['subject'] in subjs:
            subjs[info['subject']] += 1
        else:
            subjs[info['subject']] = 1

    subj_array = [(k,subjs[k]) for k in list(subjs)]
    subj_array.sort(key=lambda x:x[1], reverse=True)
    return subj_array


# sort eid and info lists by when the experiment happened
def order_eids(eids, sinfos):
    start_times = [s['start_time'] for s in sinfos]
    eids_ordered, sinfos_ordered = zip(*[(x,y) for _,x,y in sorted(zip(start_times, eids, sinfos), key=lambda z:z[0])])

    return eids_ordered, sinfos_ordered


# download and load relevant data
def load_data(eid, d_types=DEFAULT_D_TYPES, cache_dir='.'):
    just_data = one.load(
        eid=eid,
        dataset_types=d_types,
        cache_dir='labs')

    # data in the useful form of type: np array
    data_dict = dict(zip(d_types, just_data))
    sess = SessionData(data_dict)
    return sess


# calculate the feedback from trials that matter, return relevant arrays
def get_session_feedback(dt, idx=None):

    # sometimes some arrays aren't in the data. that's ok, just skip and return the error
    try:
        # turns the contrastLeft and contrastRight into usable np array
        # contrast is increased by 1 to differentiate positive / negative trials
        contrasts = (np.nan_to_num(dt.contrast_left + 1) - np.nan_to_num(dt.contrast_right + 1))
        # cxcs: contrasts x choice
        cxcs = contrasts * dt.choice

        # zero contrast trials don't count toward actual accuracy
        zero_contrasts = np.abs(contrasts) == 1

        # # if the animal doesn't make a choice then it's wrong
        # cxcs[cxcs == 0] = -1

        # only include feedback from trials that are both included and have nonzero contrast
        feedback_cut = dt.feedback[dt.included * (1 - zero_contrasts) > 0]
        cxcs_cut = cxcs[dt.included * (1 - zero_contrasts) > 0]

        # make sure feedback is 1 if choice is correct, and vice versa
        assert np.count_nonzero(np.sign((cxcs_cut-eps) * feedback_cut) - 1) == 0

    except TypeError as e:
        if idx is not None:
            print(f'eid {idx}: skipping, error happened:')
        else:
            print('skipping, error happened:')
        print(repr(e))
        return e

    return cxcs_cut, feedback_cut


# get number of of zeros made
def get_zero_counts(cxcs_cut):
    return sum(cxcs_cut == 0)


# just get the overall performance per eid for given list of eids
def get_perf(eids, sinfos):

    outs = []
    for i,eid in enumerate(eids):

        data = load_data(eid)
        tmp = get_session_feedback(data, idx=i)
        if type(tmp) == TypeError:
            continue
        cxcs, feedback = tmp
        perf = np.sum(feedback == 1) / feedback.size
        
        zeros = get_zero_counts(cxcs)
        outs.append((i, eid, perf, zeros))
        print(f'eid {i}: {eid}, perf: {perf}, zeros: {zeros}')

    return outs


# get windowed performance across all sessions
def all_window_perfs(eids, sinfos, window=20):
    min_len = window + 1

    outs = []
    for i,eid in enumerate(eids):
        data = load_data(eid)
        tmp = get_session_feedback(data, idx=i)
        if type(tmp) == TypeError:
            continue
        cxcs, feedback = tmp

        if feedback.size < min_len:
            continue

        feedback = (feedback + 1) / 2


        perfs = []
        # initial windowed performance
        running_perf = sum(feedback[:window]) / window
        for j in range(window, feedback.size):
            running_perf += (feedback[j] - feedback[j - window]) / window
            perfs.append(running_perf)

        outs.append((i, eid, perfs))
        print(f'Finished eid {i}: {eid}')

    return outs

# get smoothed performance across all sessions
def all_smoothed_perfs(eids, sinfos, a=0.5, fil=None):
    if fil is None:
        x = (1 - a) / (1 + a) # did a bunch of math on the whiteboard to figure this out
        fil_len = 111
        fil_cen = int((fil_len - 1) / 2)
        fil = np.ones((fil_len))
        fil[fil_cen] = x
        for i in range(fil_cen):
            fil[fil_cen-1-i] = fil[fil_cen-i] * a
            fil[fil_cen+1+i] = fil[fil_cen+i] * a
    else:
        assert abs(sum(fil) - 1) < eps
        fil_len = len(fil)

    outs = []
    for i,eid in enumerate(eids):
        data = load_data(eid)
        tmp = get_session_feedback(data, idx=i)
        if type(tmp) == TypeError:
            continue
        cxcs, feedback = tmp

        if feedback.size < fil_len * 2:
            continue

        feedback = (feedback + 1) / 2

        smoothed = np.convolve(feedback, fil, mode='valid')

        outs.append((i, eid, smoothed))
        print(f'Finished eid {i}: {eid}')


    return outs

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

    lab = 'hoferlab'
    cache = 'cache'
    figures = 'figures'


    setting = "all_smoothed_perfs"

    # if setting == 'windowed_perfs':
    #     subject = 'SWC_001'
    #     c_prefix = os.path.join(cache, f'{lab}-{subject}')

    #     eids, sinfos = one.search(lab=lab, details=True, subject=subject)
    #     eids, sinfos = order_eids(eids, sinfos)

    #     outs = sess_window_perfs(eids, sinfos)
    #     rows = 5
    #     cols = 5
    #     fig, ax = plt.subplots(nrows=rows, ncols=cols, sharey=True, squeeze=True)
    #     for r in range(rows):
    #         for c in range(cols):
    #             ax[r, c].plot(outs[rows*r+c][2])
    #             ax[r, c].set_title(outs[rows*r+c][0])
    #             ax[r, c].set_ylim([0, 1])

    #     plt.gcf()
    #     plt.show()

    if setting == 'all_window_perfs':

        window = 10
        # use all subjects for one lab
        eids, sinfos = one.search(lab=lab, details=True)
        subjects = get_subject_counts(sinfos)[:6]
        subjects, counts = zip(*subjects)

        perf_outs = []
        fig, ax = plt.subplots(nrows=6, ncols=1, sharey=True, squeeze=False, figsize=(20,12))

        for ind, subject in enumerate(subjects):
            print(f'Starting on subject: {subject}')
            c_prefix = f'{cache}/{lab}-{subject}'
            

            eids, sinfos = one.search(lab=lab, details=True, subject=subject)
            eids, sinfos = order_eids(eids, sinfos)
        
            # if performances are already done no point recomputing
            f_name = f'{c_prefix}-window-{window}.pkl'
            if not os.path.isfile(f_name):
                print("Stored file doesn't exist; downloading and calculating.")
                outs = all_window_perfs(eids, sinfos, window=window)
                with open(f_name, 'wb') as f:
                    pkl.dump(outs, f)
            else:
                print("Stored file exists; loading and moving on to the next subject.")
                with open(f_name, 'rb') as f:
                    outs = pkl.load(f)

            perf_outs.append((subject, outs))

            inds, eids, perfs = list(zip(*outs))
            perf_long = []
            perf_lens = []
            for k in perfs:
                perf_long += k
                perf_lens.append(len(perf_long))
            
            row, col = divmod(ind, 1)
            ax[row,col].plot(perf_long, lw=0.5)
            ax[row,col].set_ylim([0, 1])
            ax[row,col].set_title(subject)
            ax[row,col].vlines(perf_lens, ymin=0, ymax=1, linestyles='dashed')
            ax[row,col].set_xlim([0,None])

        # plot all the subjects in different plots
        # inds, eids, perfs = list(zip(*outs))
        f_prefix = f'{figures}/{lab}'
        plt.gcf()
        plt.savefig(f'{f_prefix}-window-{window}.jpg', dpi=500)
        plt.show()

    if setting == 'all_smoothed_perfs':

        a = 0.95
        a = 'custom'
        fil = np.asarray([1/6, 1/3, 1/3, 1/6])

        # use all subjects for one lab
        eids, sinfos = one.search(lab=lab, details=True)
        subjects = get_subject_counts(sinfos)[:1]
        subjects, counts = zip(*subjects)

        perf_outs = []
        fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, squeeze=False, figsize=(20,12))

        for ind, subject in enumerate(subjects):
            print(f'Starting on subject: {subject}')
            c_prefix = f'{cache}/{lab}-{subject}'
            

            eids, sinfos = one.search(lab=lab, details=True, subject=subject)
            eids, sinfos = order_eids(eids, sinfos)
        
            # if performances are already done no point recomputing
            f_name = f'{c_prefix}-smoothed-{a}.pkl'
            if not os.path.isfile(f_name):
                print("Stored file doesn't exist; downloading and calculating.")
                outs = all_smoothed_perfs(eids, sinfos, a=a, fil=fil)
                with open(f_name, 'wb') as f:
                    pkl.dump(outs, f)
            else:
                print("Stored file exists; loading and moving on to the next subject.")
                with open(f_name, 'rb') as f:
                    outs = pkl.load(f)

            perf_outs.append((subject, outs))

            inds, eids, perfs = list(zip(*outs))
            perf_lens = []
            running_len = 0
            for k in perfs:
                running_len += k.size
                perf_lens.append(running_len)

            perf_long = np.concatenate(perfs, axis=0)
            
            row, col = divmod(ind, 1)
            ax[row,col].plot(perf_long, lw=0.5)
            ax[row,col].set_ylim([0, 1])
            ax[row,col].set_title(subject)
            ax[row,col].vlines(perf_lens, ymin=0, ymax=1, linestyles='dashed')
            ax[row,col].set_xlim([0,None])

        # plot all the subjects in different plots
        # inds, eids, perfs = list(zip(*outs))
        f_prefix = f'{figures}/{lab}'
        plt.gcf()
        plt.savefig(f'{f_prefix}-smoothed-{a}.jpg')
        plt.show()


    if setting == 'all_perfs':

        eids, sinfos = one.search(lab=lab, details=True)
        subjects = get_subject_counts(sinfos)
        subjects, counts = zip(*subjects[:6])

        perf_outs = []
        fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True)

        for ind, subject in enumerate(subjects):
            print(f'Starting on subject: {subject}')
            c_prefix = f'{cache}/{lab}-{subject}'

            eids, sinfos = one.search(lab=lab, details=True, subject=subject)
            eids, sinfos = order_eids(eids, sinfos)
        
            # if performances are already done no point recomputing
            f_name = c_prefix + '-perf.pkl'
            if not os.path.isfile(f_name):
                print("Stored file doesn't exist; downloading and calculating.")
                outs = get_perf(eids, sinfos)
                with open(f_name, 'wb') as f:
                    pkl.dump(outs, f)
            else:
                print("Stored file exists; loading and moving on to the next subject.")
                with open(f_name, 'rb') as f:
                    outs = pkl.load(f)

            perf_outs.append((subject, outs))

            inds, eids, perfs, _ = list(zip(*outs))
            row, col = divmod(ind, 3)
            ax[row,col].plot(inds, perfs, 'r.-')
            ax[row,col].set_ylim([0, 1])
            ax[row,col].set_title(subject)
            ax[row,col].grid(b=True, which='major', axis='x')
            ax[row,col].set_xlabel('session EID')
            ax[row,col].set_ylabel('session performance')

        # plot all the subjects in different plots
        # inds, eids, perfs = list(zip(*outs))
        f_prefix = f'{figures}/{lab}'
        plt.gcf()
        plt.savefig(f'{f_prefix}-perf.jpg')
        plt.show()

    if setting == 'sess_perfs':

        # get all subjects with at least 15 sessions
        subject = 'SWC_002'

        print(f'Starting on subject: {subject}')
        c_prefix = f'{cache}/{lab}-{subject}'

        eids, sinfos = one.search(lab=lab, details=True, subject=subject)
        eids, sinfos = order_eids(eids, sinfos)
    
        # if performances are already done no point recomputing
        f_name = f'{c_prefix}-perf.pkl'
        if not os.path.isfile(f_name):
            print("Stored file doesn't exist; downloading and calculating.")
            outs = get_perf(eids, sinfos)
            with open(f_name, 'wb') as f:
                pkl.dump(outs, f)
        else:
            print("Stored file exists; loading and moving on to the next subject.")
            with open(f_name, 'rb') as f:
                outs = pkl.load(f)

        # plot all the subjects in different plots
        inds, eids, perfs, zeros = list(zip(*outs))
        plt.plot(inds, perfs, 'ro-')
        for j in range(len(outs)):
            plt.annotate(zeros[j], (inds[j], perfs[j]))
        plt.grid(b=True, which='major', axis='x')
        plt.title(subject)
        plt.xlabel('session EID')
        plt.ylabel('session performance')
        plt.axis([None,None,0,1])
        f_prefix = f'{figures}/{lab}-{subject}'
        plt.savefig(f'{f_prefix}-perf.jpg')
        plt.show()
        
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
