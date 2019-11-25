import os
import errno
import six
import json
import pickle
import numpy as np

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

# https://github.com/gyyang/multitask
def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


# replace value with default if equal to None
def default_for(var, val):
    if var is not None:
        return var
    return val

# one hot vector
def one_hot(length, id):
    v = np.zeros(length)
    v[id] = 1
    return v

def angle(num, unit):
    return (2 * np.pi) * (unit / num)

# choose indices at random to break up a list of length length into num pieces
def choose_index_breaks(length, num):
    rn = np.random.choice(range(1, length-1), size=num-1, replace=False)
    rn = np.concatenate((np.array([0]), np.sort(rn), np.array([length])))

    return rn