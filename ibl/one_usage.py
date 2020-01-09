"""
make sure you're in the right conda env!

ibl docs:
https://docs.internationalbrainlab.org/en/latest/_autosummary/oneibl.one.html

"""

import pickle as pkl
import os
import itertools

import errno

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

# https://github.com/gyyang/multitask
def mkdir_p(path):
    """
    Portable mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class SessionData():
    def __init__(self, data, err=True):
        self._choice = data['trials.choice']
        self._contrast_left = data['trials.contrastLeft']
        self._contrast_right = data['trials.contrastRight']
        self._feedback = data['trials.feedbackType']
        self._included = data['trials.included']

        if err:
            if self._choice is None:
                raise KeyError("choice array missing")
            if self._contrast_left is None or self._contrast_right is None:
                raise KeyError("contrast array(s) missing")
            if self._feedback is None:
                raise KeyError("feedback array missing")

        # turns the contrastLeft and contrastRight into usable np array
        # contrast is increased by 1 to differentiate positive / negative trials
        self._contrasts = np.nan_to_num(self._contrast_left) - np.nan_to_num(self._contrast_right)
        self._contrasts_adj = np.nan_to_num(self._contrast_left + 1) - np.nan_to_num(self._contrast_right + 1)
        self._zero_contrasts = (self._contrasts == 0) * 1
        self._cxc = self._contrasts_adj * self._choice

        # create default mask
        self._mask = self.set_mask(True, True)

        # make sure feedback is 1 if choice is correct, and vice versa
        assert np.count_nonzero(np.sign((self.cxc-eps) * self.feedback) - 1) == 0


    # to mask trials that aren't included and/or trials with zero contrast given
    def set_mask(self, included, zeros):
        mask = np.ones_like(self._choice)
        if included and self._included is not None:
            mask *= (self._included * 1)
        if zeros:
            # zero contrast trials don't count toward actual accuracy
            mask *= (1 - self._zero_contrasts)
        self._mask = mask

    @property
    def choice(self):
        return self._choice[self._mask == 1]
    
    @property
    def feedback(self):
        return self._feedback[self._mask == 1]

    @property
    def cxc(self):
        # cxcs: contrasts x choice
        return self._cxc[self._mask == 1]

    @property
    def contrasts(self):
        return self._contrasts[self._mask == 1]
    

    def zero_counts(self):
        return np.sum(self._zero_contrasts)

    
    
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
def load_data(eid, d_types=DEFAULT_D_TYPES, cache_dir='labs'):
    just_data = one.load(
        eid=eid,
        dataset_types=d_types,
        cache_dir=cache_dir)

    # data in the useful form of type: np array
    data_dict = dict(zip(d_types, just_data))
    sess = SessionData(data_dict)
    return sess

def report_error(e, idx=None):
    if idx is not None:
        print(f'eid {idx}: skipping, error happened:')
    else:
        print('skipping, error happened:')
    print(repr(e))


# just get the overall performance per eid for given list of eids
def get_perfs(eids):

    outs = []
    for i,eid in enumerate(eids):
        
        try:
            dt = load_data(eid)
            feedback = dt.feedback
            cxc = dt.cxc
            zeros = dt.zero_counts()
        except KeyError as e:
            report_error(e, idx=i)

        perf = np.sum(feedback == 1) / feedback.size
        
        outs.append((i, eid, perf, zeros))
        print(f'eid {i}: {eid}, perf: {perf}, zeros: {zeros}')

    return outs


# get windowed performance across all sessions
# get rid of this eventually as it generalizes to a smoothed perf with a custom window filter
def all_window_perfs(eids, sinfos, window=20):
    outs = []
    for i,eid in enumerate(eids):
        try:
            dt = load_data(eid)
            feedback = dt.feedback
            cxc = dt.cxc

            if feedback.size < window + 1:
                continue

            feedback = (feedback + 1) / 2

            perfs = []
            # initial windowed performance
            running_perf = sum(feedback[:window]) / window
            for j in range(window, feedback.size):
                running_perf += (feedback[j] - feedback[j - window]) / window
                perfs.append(running_perf)

            outs.append((i, eid, perfs))
            print(f'eid {i}: {eid}')

        except KeyError as e:
            report_error(e, idx=i)

    return outs

# get smoothed performance across all sessions
def all_smoothed_perfs(eids, sinfos, fil=None, a=0.75):
    # no provided filter
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
        try:
            dt = load_data(eid)
            feedback = dt.feedback
            cxc = dt.cxc

            if feedback.size < fil_len * 2:
                continue

            feedback = (feedback + 1) / 2

            smooth = np.convolve(feedback, fil, mode='valid')

            outs.append((i, eid, smooth))
            print(f'eid {i}: {eid}')

        except KeyError as e:
            report_error(e, idx=i)

    return outs

def sess_data(eids):
    sdata = []
    for i,eid in enumerate(eids):
        
        try:
            dt = load_data(eid)
            dt.set_mask(included=True, zeros=False)
            choice = dt.choice
            contrasts = dt.contrasts
            feedback = dt.feedback

            sdata.append((i, eid, choice, contrasts, feedback))
            print(f'eid {i}: {eid}')
        except KeyError as e:
            report_error(e, idx=i)

    return sdata


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


    setting = "sess_data"

    if setting == 'sess_data':
        # get all subjects with at least 15 sessions
        subject = 'SWC_001'
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
            #plt.scatter(np.arange(correct.size), correct, marker='o', s=64, edgecolors=rgba_edgecolors, color=rgba_colors)

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

                # wrong = ((feedback == -1) * (contrasts != 0)).nonzero()
                #plt.vlines(wrong,-1,1,linewidths=0.5,colors='r',linestyles='dotted')
                #plt.scatter(np.arange(correct.size), correct, marker='o', s=49)
                #plt.plot(correct, 'r:', lw=0.5)
                #plt.plot(choices, 'b:', lw=0.5)
                # for i in range(feedback.size):
                #     if contrasts[i] > 0:
                #         #plt.axvspan(i-.5, i+.5, facecolor='g', alpha=abs(contrasts[i]) / 2)
                #         plt.plot(i, 1.3, marker='s', color='black', ms=5)
                #     if contrasts[i] < 0:
                #         #plt.axvspan(i-.5, i+.5, facecolor='r', alpha=abs(contrasts[i]) / 2)
                #         plt.plot(i, -1.3, marker='s', color='black', ms=5)
                #plt.plot(feedback[-s][-150:], 'g-')
                
            #fig.suptitle(f'{subject}, session {s}')
            fig.tight_layout()
            
            f_name = f'{fig_path}/{inds[s]}-data.jpg'
            plt.savefig(f_name)
            print(f"Processed session {inds[s]}")
            plt.close()
            #pdb.set_trace()
        #plt.show()


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
                outs = get_perfs(eids)
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
            outs = get_perfs(eids, sinfos)
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
