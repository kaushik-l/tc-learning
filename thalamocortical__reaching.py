import torch
import math
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
import model

# thalamocortical model for reaching

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


def simulate_teacher(params, plant, xtarg, xgo):
    xtarg, xgo = torch.as_tensor(xtarg), torch.as_tensor(xgo)
    # random initialization of hidden state
    h = torch.as_tensor(params['sig'] * npr.randn(1, N).astype(np.float32))
    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)
    # initial noise
    noise = torch.as_tensor(npr.randn(1, R).astype(np.float32)) * plant.noise_scale
    # initial arm pos and velocity in angular coordinates
    w = torch.as_tensor(np.tile(plant.w_init, (1, 1)), dtype=torch.float32)
    v = torch.zeros(1, R)
    # initial output
    ua = torch.zeros(NT, R)  # angular acceleration of joints
    na = torch.as_tensor(npr.randn(NT, R).astype(np.float32)) * plant.noise_scale  # multiplicative noise
    xa = torch.zeros(NT, R)  # angular position of joints
    for ti in range(NT):
        # network update
        s = torch.cat((xtarg, xgo[ti]), dim=-1)[None, :] # input
        h = h + dt * (-h + torch.tanh(s.float().mm(params['ws']) + h.mm(params['J']) + params['b']))  # dynamics
        u = h.mm(params['wr'])  # output
        # physics
        noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
        v, w, x = plant.forward(u, v, w, noise, dt)  # actual state
        # save values for plotting
        ua[ti], na[ti], xa[ti] = u, noise, x
    return ua.detach().numpy(), xa.detach().numpy()


# load data
params = {'wr': [], 'ws': [], 'b': [], 'J': [], 'sig': []}
teacher = torch.load('bptt__reaching.pt')['net']
plant = torch.load('bptt__reaching.pt')['plant']
(params['wr'], params['ws'], params['J'], params['b'], params['sig']) = (
    teacher.wr.clone().detach(), teacher.ws.clone().detach(), teacher.J.clone().detach(), teacher.b.clone().detach(), teacher.sig)
del teacher


# initialize network parameters
tc_type = 'readout'                                 # 'random', 'readout' or 'pc'
S, N, M, R, g, dt = 3, 200, 10, 2, 1.5, .1
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
angles = math.pi/4 * np.arange(8)                   # 8 reaching angles
g_lims = (10, 15)
duration, cycles = 35, 2
NT = int(duration / dt)
t = dt * np.arange(NT)
s = 0.0 * np.ones((S, NT))

# learning algorithm parameters
lr, Nepochs = 0.05, 10000

# track variables during learning
learning = {'epoch': [], 'mses': [], 'fbalignment': []}

# save tensors for plotting
sa = np.zeros((NT, S))  # input
ha = np.zeros((NT, N))  # ctx activity
ra = np.zeros((NT, M))  # thal activity
ua = np.zeros((NT, R))  # output motor command
xa = np.zeros((NT, R))  # output position
na = torch.as_tensor(npr.randn(NT, R).astype(np.float32)) * plant.noise_scale

# start training using RFLO
for ei in range(Nepochs):

    # initial activity
    z, h, v, r = z0, h0, v0, r0

    # error
    err = np.zeros((NT, R))

    # eligibility trace
    q = np.zeros((N, M))

    # pick a reach location and delay
    ang = angles[npr.randint(len(angles))]
    xtarg = np.array([np.cos(ang), np.sin(ang)+1])        # reach location
    start = np.round((npr.random(1) * (g_lims[1] - g_lims[0]) + g_lims[0]) / dt).squeeze().astype(int)
    xgo = np.concatenate((np.ones((start, 1)), np.zeros((NT - start, 1))))
    # simulate teacher
    ustar, xstar = simulate_teacher(params, plant, xtarg, xgo)

    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)
    # initial noise
    noise = torch.as_tensor(npr.randn(1, R).astype(np.float32)) * plant.noise_scale
    # initial arm pos and velocity in angular coordinates
    w = torch.as_tensor(np.tile(plant.w_init, (1, 1)), dtype=torch.float32)
    vel = torch.zeros(1, R)

    for ti in range(NT):

        # update membrane potential
        Iin = np.matmul(ws, np.concatenate((xtarg, xgo[ti])))[:, None]
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
        err[ti] = ustar[ti] - u.flatten()         # error

        # update readout and thalamocortical weights
        wr += (((lr / NT) * h) * err[ti]).T
        wct += ((lr / NT) * np.matmul(B, err[ti]).reshape(N, 1) * q)

        # physics
        noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
        vel, w, x = plant.forward(torch.as_tensor(u.T), vel, w, noise, dt)  # actual state

        # save values for plotting
        ha[ti], ra[ti], ua[ti], xa[ti] = h.T, r.T, u.T, x.detach().numpy()

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
    print('\r' + str(ei) + '/' + str(Nepochs) + '\t Err:' + str(mse), end='')
    if ei % 100 == 0:
        fig = plt.figure(figsize=(12, 12))
        plt.subplot(221)
        plt.plot(ustar, 'k'), plt.ylim((-0.2, 0.2))
        plt.title('Target motor command')
        plt.subplot(222)
        plt.plot(xstar[:,0], xstar[:,1], 'k'), plt.plot(xtarg[0], xtarg[1], 'ok')
        plt.xlim((-1.1, 1.1)), plt.ylim((0, 2.2))
        plt.title('Target trajectory')
        plt.subplot(223)
        plt.plot(ua, 'r'), plt.ylim((-0.2, 0.2))
        plt.title('Model motor command')
        plt.subplot(224)
        plt.plot(xa[:,0], xa[:,1], 'r'), plt.plot(xtarg[0], xtarg[1], 'ok')
        plt.xlim((-1.1, 1.1)), plt.ylim((0, 2.2))
        plt.title('Model trajectory')
        plt.xlabel('Time', fontsize=18), plt.ylabel('Output', fontsize=18)
        plt.suptitle('Epoch ' + str(ei) + '/' + str(Nepochs))
        plt.tight_layout()
        plt.show()

    # compute overlap and save
    learning['fbalignment'].append((wr.flatten() @ B.flatten('F')) /
                                   (np.linalg.norm(wr.flatten()) * np.linalg.norm(B.flatten('F'))))
    learning['epoch'].append(ei)
    learning['mses'].append(mse)


# plot learning curve and output
plt.close()
fig = plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.plot(ustar, 'k')
plt.title('Target motor command')
plt.subplot(222)
plt.plot(xstar[:, 0], xstar[:, 1], 'k'), plt.plot(xtarg[0], xtarg[1], 'ok')
plt.xlim((-1.1, 1.1)), plt.ylim((0, 2.2))
plt.title('Target trajectory')
plt.subplot(223)
plt.plot(ua, 'r')
plt.title('Model motor command')
plt.subplot(224)
plt.plot(xa[:, 0], xa[:, 1], 'r'), plt.plot(xtarg[0], xtarg[1], 'ok')
plt.xlim((-1.1, 1.1)), plt.ylim((0, 2.2))
plt.title('Model trajectory')
plt.xlabel('Time', fontsize=18), plt.ylabel('Output', fontsize=18)
plt.suptitle('Epoch ' + str(ei) + '/' + str(Nepochs))
plt.tight_layout()
plt.show()
# plt.pause(5)
