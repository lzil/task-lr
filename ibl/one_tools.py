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


DEFAULT_D_TYPES = [
    'trials.choice',
    'trials.contrastLeft',
    'trials.contrastRight',
    'trials.feedbackType',
    'trials.choice',
    'trials.included'
]
    

# custom data class that takes care of included and zero contrast trials
class SessionData():
    def __init__(self, data, err=True):
        self._choice = data['trials.choice']
        self._contrast_left = data['trials.contrastLeft']
        self._contrast_right = data['trials.contrastRight']
        self._feedback = data['trials.feedbackType']
        self._included = data['trials.included']

        self._rtimes = None
        self._gtimes = None
        self._intervals = None
        self.wp = None
        self.wt = None

        if 'trials.response_times' in data:
            self._rtimes = data['trials.response_times']
            
        if 'trials.goCue_times' in data:
            self._gtimes = data['trials.goCue_times']
            
        if 'trials.intervals' in data:
            self._intervals = data['trials.intervals']
            

        if 'wheel.position' in data:
            self.wp = data['wheel.position']
            
        if 'wheel.timestamps' in data:
            self.wt = data['wheel.timestamps']
            

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
        self.set_mask(True, True)

        self.session_len = np.count_nonzero(self._mask)

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

    def wheel_indices(self):
        assert self.wp is not None
        assert self.wt is not None
        assert self._intervals is not None
        w_ind = 0
        groups = []
        cur_group = []
        for i in range(len(self._choice)):
            while w_ind < len(self.wt) and self.wt[w_ind] < self._rtimes[i]:#self._intervals[i,1]:
                if self.wt[w_ind] >= self._gtimes[i]:
                    cur_group.append(w_ind)
                w_ind += 1
            groups.append((i,cur_group))
            cur_group = []

        # only take those groups not in the mask
        #groups = list(filter(lambda x: self._mask[x[0]] == 1, groups))

        return groups

    @property
    def rtimes(self):
        return self._rtimes[self._mask == 1]

    @property
    def gtimes(self):
        return self._gtimes[self._mask == 1]
    
    @property
    def intervals(self):
        return self._intervals[np.where(self._mask == 1)[0],:]

    def zero_counts(self):
        return np.sum(self._zero_contrasts)


def report_error(e, idx=None):
    if idx is not None:
        print(f'eid {idx}: skipping, error happened:')
    else:
        print('skipping, error happened:')
    print(repr(e))



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
def load_data(one, eid, d_types=DEFAULT_D_TYPES, cache_dir='labs', sess_type=True, err=True):
    just_data = one.load(
        eid=eid,
        dataset_types=d_types,
        cache_dir=cache_dir)

    # data in the useful form of type: np array
    data_dict = dict(zip(d_types, just_data))
    if not sess_type:
        return data_dict
    sess = SessionData(data_dict, err)
    return sess



def load_maybe_save(one, lab, subject, dtypes=DEFAULT_D_TYPES, base_dir='cache'):
    cache_path = f'{base_dir}/{lab}/{subject}'
    mkdir_p(cache_path)

    # if performances are already done no point recomputing
    f_name = f'{cache_path}/data.pkl'
    if not os.path.isfile(f_name):

        eids, sinfos = one.search(lab=lab, details=True, subject=subject)
        eids, sinfos = order_eids(eids, sinfos)

        dts = []
        print(f'Gathering {lab}/{subject}...')
        for i, eid in enumerate(eids):
            try:
                dt = load_data(one, eid, d_types=d_types, err=True)
                dts.append((i, eid, sinfos[i], dt))
            except KeyError as e:
                report_error(e, idx=i)
        print(f'Finished {lab}/{subject}.')

        with open(f_name, 'wb') as f:
            pkl.dump(dts, f)

    else:
        with open(f_name, 'rb') as f:
            dts = pkl.load(f)

    return dts