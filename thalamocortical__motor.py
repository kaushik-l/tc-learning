import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr

# thalamocortical model for motor learning

# seed
npr.seed(1)


# non-linearity
def f(x):
    return np.tanh(x)


# derivative of non-linearity
def df(x):
    return 1 - np.tanh(x) ** 2


def loss(err):
    mse = (err ** 2).mean() / 2
    return mse


# initialize network parameters
tc_type = 'readout'                                 # 'random', 'readout' or 'pc'
S, N, M, R, g, dt = 1, 256, 1, 1, 1.5, .1
ws = (2 * npr.random((N, S)) - 1) / np.sqrt(S)      # input weights
wcc = g * npr.standard_normal([N, N]) / np.sqrt(N)  # corticocortical weights (ctx -> ctx)
wct = g * npr.standard_normal([N, M]) / np.sqrt(M)  # thalamocortical weights (thal -> ctx)
wr = (2 * npr.random((R, N)) - 1) / np.sqrt(N)      # readout weights
wtc = g * npr.standard_normal([M, N]) / np.sqrt(N)  # corticothalamic weights (ctx -> thal)
B = npr.standard_normal([N, R]) / np.sqrt(R)        # feedback weights
z0 = npr.randn(N, 1)                                # ctx potential
h0 = f(z0)                                          # ctx activity
v0 = np.matmul(wtc, h0)                             # thal potential
r0 = f(v0)                                          # thal activity
pca = PCA(n_components=M)                           # calculate top M PCs of ctx if needed

# task parameters
duration, cycles = 20, 2
NT = int(duration / dt)
t = dt * np.arange(NT)
s = 0.0 * np.ones((S, NT))
ustar = (np.sin(2 * np.pi * np.arange(NT) * cycles / (NT - 1)) +
         0.75 * np.sin(2 * 2 * np.pi * np.arange(NT) * cycles / (NT - 1)) +
         0.5 * np.sin(4 * 2 * np.pi * np.arange(NT) * cycles / (NT - 1)) +
         0.25 * np.sin(6 * 2 * np.pi * np.arange(NT) * cycles / (NT - 1)))
ustar /= np.std(ustar)

# learning algorithm parameters
lr, Nepochs = 0.1, 10000

# track variables during learning
learning = {'epoch': [], 'mses': [], 'fbalignment': []}

# save tensors for plotting
sa = np.zeros((NT, S))  # input
ha = np.zeros((NT, N))  # ctx activity
ra = np.zeros((NT, M))  # thal activity
ua = np.zeros((NT, R))  # output

# plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
line1, = ax.plot(ustar, 'k')
line2, = ax.plot(ua, 'r')
plt.xlabel('Time', fontsize=18), plt.ylabel('Output', fontsize=18)
fig.show(), fig.legend({'Model', 'Target'}, loc='upper right', fontsize=18)

# start training using RFLO
for ei in range(Nepochs):

    # initial activity
    z, h, v, r = z0, h0, v0, r0

    # error
    err = np.zeros((NT, R))

    # eligibility trace
    q = np.zeros((N, M))

    for ti in range(NT):

        # update membrane potential
        Iin = np.matmul(ws, s[:, ti])[:, None]
        Irec = np.matmul(wcc, h)
        Ict = np.matmul(wct, r)
        z = Iin + Irec + Ict

        # update eligibility trace
        q = dt * df(z) * r.T + (1 - dt) * q

        # update output and error
        h = (1 - dt) * h + dt * f(z)    # ctx activity
        v = np.matmul(wtc, h)           # thal potential
        r = f(v)                        # thal activity
        u = np.matmul(wr, h)            # output
        err[ti] = ustar[ti] - u         # error

        # update readout and thalamocortical weights
        wr += (((lr / NT) * h) * err[ti]).T
        wct += ((lr / NT) * np.matmul(B, err[ti]).reshape(N, 1) * q)

        # save values for plotting
        ha[ti], ra[ti], ua[ti] = h.T, r.T, u.T

    # update corticothalamic weights if needed
    lr_tc = 1e-2
    if tc_type == 'readout':
        wtc[:R, :] = (1 - lr_tc) * wtc[:R, :] + lr_tc * wr[:R, :]
    elif tc_type == 'pc':
        pca.fit(ha)
        pcs = pca.components_ / np.sqrt(pca.singular_values_[:, None])
        sign = np.sign([pearsonr(wtc[idx, :], pcs[idx, :])[0] for idx in range(M)])
        lr_tc = lr_tc / (np.arange(1, M + 1) ** 2)
        wtc = (1 - lr_tc[:, None]) * wtc[:, :] + lr_tc[:, None] * pcs * sign[:, None]

    # print loss and update plot
    mse = loss(err)
    if ei % 100 == 0:
        print('\r' + str(ei) + '/' + str(Nepochs) + '\t Err:' + str(mse), end='')
        line2.set_ydata(ua)
        fig.canvas.draw()
        plt.title('Epoch ' + str(ei) + '/' + str(Nepochs))
        plt.pause(0.0001)

    # compute overlap and save
    learning['fbalignment'].append((wr.flatten() @ B.flatten('F')) /
                                   (np.linalg.norm(wr.flatten()) * np.linalg.norm(B.flatten('F'))))
    learning['epoch'].append(ei)
    learning['mses'].append(mse)


# plot learning curve and output
plt.close()
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
plt.plot(learning['mses'], 'r')
plt.ylim((1e-3, 1))
plt.yscale('log')
plt.xlabel('Epoch', fontsize=18), plt.ylabel('Error', fontsize=18)
ax2 = fig.add_subplot(132)
plt.plot(np.abs(learning['fbalignment']), 'r')
plt.ylim((0, 1))
plt.xlabel('Epoch', fontsize=18), plt.ylabel('Feedback alignment', fontsize=18)
ax3 = fig.add_subplot(133)
line1, = ax3.plot(ustar, 'k')
line2, = ax3.plot(ua, 'r')
plt.xlabel('Time', fontsize=18), plt.ylabel('Output', fontsize=18)
ax3.legend({'Model', 'Target'}, loc='upper right', fontsize=18)
plt.tight_layout()
fig.show()
# plt.pause(5)
