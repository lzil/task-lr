import numpy as np
import pdb
import os
import matplotlib.pyplot as plt


from task_lr import *
record_interval = 100
record_interval = 1
num_reps = 20

gamma = 1#.7


def single_trial(N, q, lr, mask=0, clip=False, iters=50000):
    # quantifying dimensionality prior relevance
    dims = range(N)
    dims_p = [1] * N
    for i in range(1,N):
        dims_p[i] = dims_p[i-1]*gamma
    dims_p = list(map(lambda x: x / sum(dims_p), dims_p))
    # choose dimensions that are actually relevant and create true weight (binary)

    # n dimensions are relevant
    # relevant_dims = np.random.choice(N, n, replace=False)
    # w_star = np.zeros(N)
    # for d in relevant_dims:
    #     w_star[d] = 1

    # each dimension is relevant with probability q
    # w_star = np.random.choice(2, N, replace=True, p=[1-q,q])

    # each dimension is relevant with prior distro
    num_relevant_dims = round(q * N)
    relevant_dims = np.random.choice(dims, size=num_relevant_dims, p=dims_p)
    w_star = np.zeros(N)
    for d in relevant_dims:
        w_star[d] = 1

    # iterate with simple learning rule
    def step(c, w):
        s_trial = sample_s(N)

        # masking
        if mask is not 0:
            p_mask = mask

            # certain probability of masked dimensions
            # z_mask = np.random.choice(2, N, replace=True, p=[p_mask,1-p_mask])

            # n_mask masked dimensions
            # n_mask = round(p_mask * N)
            # relevant_dims = np.random.choice(N, n_mask, replace=False)
            # z_mask = np.ones(N)
            # for d in relevant_dims:
            #    z_mask[d] = 0

            # prior probability distro over relevant dims
            n_mask = round(p_mask * N)
            relevant_dims = np.random.choice(dims, size=n_mask, p=dims_p)
            z_mask = np.ones(N)
            for d in relevant_dims:
                z_mask[d] = 0
        else:
            z_mask = np.ones(N)

        

        r_diff = (w_star - w) @ s_trial

        se = 1/N * np.linalg.norm(r_diff) ** 2
        w_delta = lr * r_diff * z_mask * s_trial

        w_new = w + w_delta

        if clip:
            w_cur = np.clip(w_cur, 0, 1)

        # if not c % 1000 and c > 0:
        #     print('Iteration {}: mse: {}'.format(c, mse))
        return w_new, se


    def train():
        w_cur = np.zeros(N)
        mse_arr = []
        print('Running: {} units learning stimuli with p={}, lr={}'.format(N, q, lr))
        for c in range(iters):
            if not c % record_interval:
                mse = 0
                for i in range(num_reps):
                    w_new, se = step(c, w_cur)
                    mse += se
                mse /= num_reps
                mse_arr.append(mse)
                w_cur = w_new
            else:
                w_cur, _ = step(c, w_cur)

        return mse_arr


    return train()

    # print('final w: {}'.format(w_cur))


mse_ideal_len = 4

lr_list = {
    10: [0.05, 0.1, 0.16, 0.2, 0.21],
    20: [0.095, 0.098, 0.1, 0.1001],
    50: [0.01, 0.02, 0.03, 0.036, 0.039, 0.04, 0.0403],
    100: [0.001, 0.01, 0.015, 0.019, 0.02, 0.021],
    200: [0.001, 0.005, 0.009, 0.01, 0.0101, 0.011],
    400: [0.001, 0.004, 0.0049, 0.005, 0.0051, 0.006]
}

Ns = [10]
q = 0.2

iterations = 50

for N in Ns:
    mse_range = []
    mse_range_ideal = []
    for lr in lr_list[N]:
        for m in [0]:
            mses = single_trial(N, q, lr, mask=m, iters=iterations)
            mse_range.append(mses)

            # mses_ideal = np.zeros(mse_ideal_len)
            # mses_ideal[0] = 1
            # lr_ideal = (1 - 2*lr + N*(lr**2)) ** 1000
            # for i in range(mse_ideal_len - 1):
            #     mses_ideal[i + 1] = mses_ideal[i] * lr_ideal
            # mse_range_ideal.append(mses_ideal)
            prefactor = np.log(1 - 2*lr + N*(lr**2))
            mses_ideal = record_interval * prefactor * np.arange(mse_ideal_len)
            mse_range_ideal.append(mses_ideal)


    mse_range = np.log(np.asarray(mse_range))
    mse_range_ideal = np.asarray(mse_range_ideal)

    cmap = plt.get_cmap('jet_r')

    for i,lr in enumerate(mse_range):
        color = cmap(float(i)/len(lr_list[N]))
        plt.plot(lr, c=color,label=lr_list[N][i])
        plt.plot(mse_range_ideal[i], c=color, ls='--')

    plt.title('learning rate vs MSEs for N={},p={},m=0.2'.format(N,q))
    plt.xlabel('iteration (x{})'.format(record_interval))
    plt.ylabel('log MSE')
    plt.legend()
    plt.show()

    save_path = os.path.join('figures', N, f'naive_sim_{N}_{q}')
    #plt.savefig('naive_sim_{}_{}_{}.png'.format(N,q,0.2))
    #plt.clf()

