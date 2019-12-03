# make sure you're in the right conda env!

from oneibl.one import ONE
from ibllib.misc import pprint
import numpy as np


def get_subject_counts(sinfos):
    subjs = {}
    for info in sinfos:
        if info['subject'] in subjs:
            subjs[info['subject']] += 1
        else:
            subjs[info['subject']] = 1
    return subjs


# sort eid and info lists by when the experiment happened
def order_by_start(eids, sinfos):
    sinfos_date_ordered,indices = zip(*sorted(list(zip(sinfos,range(len(sinfos)))), key=(lambda x: x[0]['start_time'])))
    eids_date_ordered,_ = zip(*sorted(list(zip(eids,indices)), key=(lambda x: x[1])))
    return eids_date_ordered, sinfos_date_ordered


if __name__ == '__main__':
    one = ONE()


    lab = 'hoferlab'
    eids, sinfos = one.search(lab=lab, details=True, subject='SWC_015')
    eids, sinfos = order_by_start(eids, sinfos)

    # session_data = one.load(eids[-3], cache_dir='.')

    # types = session_data.dataset_type
    # dt = session_data.data

    outs = []
    for i,eid in enumerate(eids):
        if 'trials.choice' in one.list(eid, keyword='dataset_type'):
            #print(f'found choice in eid {i}: {eid}')
            left, right, fback, choice, pleft, rnum = one.load(
                eid,
                cache_dir='.',
                dataset_types=[
                    'trials.contrastLeft',
                    'trials.contrastRight',
                    'trials.feedbackType',
                    'trials.choice',
                    'trials.probabilityLeft',
                    'trials.repNum'
                ])

            # turns the contrastLeft and contrastRight into usable np array.
            both = (np.nan_to_num(left + 1) - np.nan_to_num(right + 1)) * choice
            both[both == 0] = -1


            # print(np.sign(both) - np.sign(fback))
            perf =  np.sum(fback > 0) / fback.size
            outs.append((i, eid, perf))
            print(f'eid {i}: {eid}, performance: {perf}')

    # print out how the mouse is performing on subsequent sessions (that have data)
    # for o in outs:
    #     print(f'eid {o[0]}: {o[1]}, performance: {o[2]}')
