import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

# number of samples for MSE calculation
n_mse_reps = 10


# no correlations; good lr choices; lr must be <2/N
lr_list = {
    10: [0.05, 0.1, 0.16, 0.2, 0.21], # 0.2
    20: [0.095, 0.098, 0.1, 0.1001], # 0.1
    50: [0.01, 0.02, 0.03, 0.036, 0.039, 0.04, 0.0403], # 0.04
    100: [0.001, 0.01, 0.015, 0.019, 0.02, 0.021], # 0.02
    200: [0.001, 0.005, 0.009, 0.01, 0.0101, 0.011], # 0.01
    400: [0.001, 0.004, 0.0049, 0.005, 0.0051, 0.006] # 0.005
}

# give back a binary sample s with n dimensions
def sample_s(n):
    return 2 * np.random.randint(0, 2, size=n) - 1


def single_trial(N, q, lr, gamma=1, m=1, clip=False, s_corr=1, iters=50000, rec_int=1000):
    """train a single trial with a single parameter setting
    
    Args:
        N (TYPE): number of dimensions
        q (float): (0,1); proportion of dims that are relevant
        lr (float): (0,1); learning rate
        gamma (int, optional): decay rate of prior
        m (float, optional): proportion of dims to pay attention to
        clip (bool, optional): whether to clip weights after every step
        iters (int, optional): how many iterations to run for
    """

    # quantifying dimensionality prior relevance, as well as for attention masking
    # falls off as [1, gamma, gamma^2, ...]
    dims = np.arange(N)
    dims_p = np.ones(N)
    for i in range(1,N):
        dims_p[i] = dims_p[i-1]*gamma
    dims_p /= np.sum(dims_p)


    # each dimension is relevant with prior distribution given by dims_p
    n_rel_dims = round(q * N)
    rel_dims = np.random.choice(dims, size=n_rel_dims, p=dims_p)

    # defining the optimal w
    w_star = np.zeros(N)
    for d in rel_dims:
        w_star[d] = 1

    # iterate with simple hebbian learning rule
    def step(w, s_prev):
        if s_corr != 0:
            n_change = round((1 - s_corr) / 2 * N)
            dims_to_flip = np.random.choice(dims, size=n_change)
            s_trial = s_prev
            s_trial[dims_to_flip] *= -1
        else:
            s_trial = sample_s(N)

        # pay attention to m proportion of dimensions
        # dimensionality also falls as a function of gamma
        z_att = np.ones(N)
        if m < 1:
            n_att = round(m * N)
            att_dims = np.random.choice(dims, size=n_att, p=dims_p)
            for d in att_dims:
                z_att[d] = 0

        # actual hebbian learning rule
        r_diff = (w_star - w) @ s_trial

        w_delta = lr * r_diff * z_att * s_trial
        w_new = w + w_delta

        # mse calculation
        se = 1/N * np.linalg.norm(r_diff) ** 2

        # clip w if it's over 1. hasn't been used in a while
        if clip:
            w_new = np.clip(w_new, 0, 1)

        return s_trial, w_new, se


    w_cur = np.zeros(N)
    mse_arr = []
    print(f'Running: {N} units learning stimuli with q={q}, lr={lr}')
    for c in range(iters):
        s = sample_s(N)
        if not c % rec_int:
            # run for some number of timesteps each to get a good MSE estimate
            # then take the last step, whichever it is
            # but ONLY on recorded steps
            mse = 0
            for i in range(n_mse_reps):
                s, w_new, se = step(w_cur,s)
                mse += se
            mse /= n_mse_reps
            mse_arr.append(mse)
            w_cur = w_new
        else:
            s, w_cur, _ = step(w_cur,s)

    return mse_arr


if __name__ == '__main__':
    

    # mse_ideal_len = 4

    q = 0.2
    gamma = 1
    n_iters = 20000
    rec_int = 100

    Ns = [50,100, 200]

    ms = [1]

    # mse_actual contains real MSEs from training
    mse_actual = {}
    # mse_theory contains theoretical learning rate bound MSEs
    mse_theory = {}

    for N in Ns:
        mse_actual[N] = {}
        mse_theory[N] = {}
        for m in ms:
            mse_actual[N][m] = []
            mse_theory[N][m] = []
            for lr in lr_list[N]:
                # actually acquire MSEs
                mses = single_trial(N, q, lr, gamma=gamma, iters=n_iters, s_corr=1, rec_int=rec_int)
                mse_actual[N][m].append(mses)

                # theoretical learning rate bound for no correlations
                prefactor = np.log(1 - 2*lr + N*(lr**2))
                mses_ideal = np.log(q) + rec_int * prefactor * np.arange(len(mses))
                mse_theory[N][m].append(mses_ideal)

            # convert to np arrays; log already taken in prefactor for theory
            mse_actual[N][m] = np.log(np.asarray(mse_actual[N][m]))
            mse_theory[N][m] = np.asarray(mse_theory[N][m])

    cmap = plt.get_cmap('jet_r')

    fig,ax = plt.subplots(nrows=len(ms),ncols=len(Ns),squeeze=False)
    for i, N in enumerate(Ns):
        for j, m in enumerate(ms):
            for k,lr in enumerate(mse_actual[N][m]):
                color = cmap(float(k)/len(lr_list[N]))
                ax[j,i].plot(lr, c=color,label=lr_list[N][k])
                ax[j,i].plot(mse_theory[N][m][k], c=color, ls='--', linewidth=1)

                ax[j,i].set_title(f'N={N}, m={m}')
                ax[j,i].set_xlabel(f'iter (x{rec_int})')
                ax[j,i].set_ylabel('log MSE')
                ax[j,i].legend()

    plt.savefig('playground/test.png')

    #save_path = os.path.join('figures', N, f'naive_sim_{N}_{q}')
    #plt.savefig('naive_sim_{}_{}_{}.png'.format(N,q,0.2))
    #plt.clf()

