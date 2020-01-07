"""
make sure you're in the right conda env!

ibl docs:
https://docs.internationalbrainlab.org/en/latest/_autosummary/oneibl.one.html

"""

import pickle as pkl
import os

from oneibl.one import ONE
from ibllib.misc import pprint

import numpy as np
import matplotlib.pyplot as plt


def get_subject_counts(sinfos):
    subjs = {}
    for info in sinfos:
        if info['subject'] in subjs:
            subjs[info['subject']] += 1
        else:
            subjs[info['subject']] = 1
    return subjs


# sort eid and info lists by when the experiment happened
def order_eids(eids, sinfos):
    start_times = [s['start_time'] for s in sinfos]
    eids_ordered, sinfos_ordered = zip(*[(x,y) for _,x,y in sorted(zip(start_times, eids, sinfos), key=lambda z:z[0])])

    return eids_ordered, sinfos_ordered

    # sinfos_date_ordered,indices = zip(*sorted(list(zip(sinfos,range(len(sinfos)))), key=(lambda x: x[0]['start_time'])))
    # eids_date_ordered,_ = zip(*sorted(list(zip(eids,indices)), key=(lambda x: x[1])))
    # return eids_date_ordered, sinfos_date_ordered

def get(dic, dt):
    if dt in dic:
        return dic[dt]
    else:
        return None


if __name__ == '__main__':
    one = ONE()

    # if performances are already done no point recomputing
    if not os.path.isfile('perf.pkl'):

        lab = 'hoferlab'
        eids, sinfos = one.search(lab=lab, details=True, subject='SWC_015')
        eids, sinfos = order_eids(eids, sinfos)

        outs = []
        for i,eid in enumerate(eids):
            # check to make sure that the animal choice is actually recorded
            if 'trials.choice' in one.list(eid, keyword='dataset_type'):

                # actually downloads the data
                whole_data = one.load(
                    eid,
                    cache_dir='.')

                # data in the useful form of type: np array
                data = dict(zip(whole_data.dataset_type, whole_data.data))

                try:
                    contrast_left = data['trials.contrastLeft']
                    contrast_right = data['trials.contrastRight']
                    feedback = data['trials.feedbackType']
                    choice = data['trials.choice']
                    p_left = data['trials.probabilityLeft']
                    included = data['trials.included']

                except KeyError as e:
                    # just move on to the next eid if essential fields are missing
                    print(repr(e))
                    continue


                # turns the contrastLeft and contrastRight into usable np array.
                both = (np.nan_to_num(contrast_left + 1) - np.nan_to_num(contrast_right + 1)) * choice
                # if the animal doesn't make a choice then it's wrong
                both[both == 0] = -1

                # zero contrast trials don't count toward actual accuracy
                zero_contrasts = np.abs(both) == 1

                # only include feedback from trials that are both included and have nonzero contrast
                feedback_cut = feedback[included * (1 - zero_contrasts) > 0]
                perf = np.sum(feedback_cut == 1) / feedback_cut.size
                outs.append((i, eid, perf))
                print(f'eid {i}: {eid}, performance: {perf}')

            else:
                print(f'eid {i}: no choice array, skipping {eid}')


        # save the performances
        with open('perf.pkl', 'wb') as f:
            pkl.dump(outs, f)

    else:

        with open('perf.pkl', 'rb') as f:
            outs = pkl.load(f)

    inds, eids, perfs = list(zip(*outs))
    plt.plot(inds, perfs)
    plt.show()

