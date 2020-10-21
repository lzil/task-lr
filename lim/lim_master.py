import sys
import os
import pdb
import datetime

import csv

import numpy as np
import matplotlib.pyplot as plt


def get_animal_counts(reader):
    fn = reader.fieldnames

    animal_counts = {}

    for row in reader:
        if row['animal_id'] not in animal_counts:
            animal_counts[row['animal_id']] = 0
        else:
            animal_counts[row['animal_id']] += 1
    

    return animal_counts


def order_sessions(rows):
    return sorted(rows, key=lambda r: datetime.datetime.strptime(r['date'], '%d-%b-%Y'))


def analyze_training():
    data_file = os.path.join('data', 'TRAINING.csv')
    
    with open(data_file, 'r') as f:

        reader = csv.DictReader(f, delimiter=",")
        fn = reader.fieldnames
        
        subject = 'AA02'

        # first find all the sessions belonging to subject mouse

        rows = []
        for row in reader:
            if row['animal_id'] == subject:
                if int(row['all_trials']) != 0:
                    rows.append(row)

    # gotta reorder by date
    rows = order_sessions(rows)

    dt = []
    ind = 0
    for row in rows:
        ind += 1
        dt.append((
            ind,
            int(row['stage']),
            int(row['all_trials']),
            int(row['correct_trials']),
            int(row['error_trials']),
            int(row['violation_trials']),
            ))



    dt = list(zip(*dt))
    inds = dt[0]
    alls = np.asarray(dt[2])
    corrects = np.asarray(dt[3])
    perfs = np.divide(corrects, alls)
    colors = np.zeros((ind, 3))
    stages = np.asarray(dt[1]) * 1
    colors[:,0] = 1 - stages
    colors[:,2] = stages

    plt.scatter(
        x = inds,
        y = perfs,
        c = colors
        )


    for j in inds:
        plt.annotate(alls[j-1], (j, perfs[j-1]), fontsize='small')

    plt.ylabel('performance')
    plt.xlabel('session')
    plt.title(subject)

    plt.show()


def analyze_trials():
    pass



if __name__ == '__main__':

    analyze_training()



    