import torch
from time import time
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from plot import plot
from model import Network, Task, Plant, Algorithm

# instantiate model
net, task, plant, algo = Network('Ctx'), Task('Reaching'), Plant('TwoLink'), Algorithm('Adam', 5000)

# seed
npr.seed(1)

# frequently used vars
dt, NT, B, N, S, R = net.dt, int(task.T / net.dt), algo.B, net.N, net.S, net.R    # single cortex
t = dt * np.arange(NT)

# OU noise parameters
ou_param_1 = np.exp(-dt / plant.noise_corr_time)
ou_param_2 = np.sqrt(1 - ou_param_1**2)

# optimizer
opt = None
if algo.name == 'Adam':
    opt = torch.optim.Adam([net.J, net.wr, net.b, net.ws], lr=algo.lr)

# figure preferences
doplot = True
num_to_plot = 4
plot_every = algo.Nepochs * 0.1
fig = plt.figure(figsize=(12, 7))

# track variables during learning
lr_ = algo.lr
losses, mses, u_regs, du_regs = [], [], [], []
prevt = time()

for ei in range(algo.Nepochs):

    # initial and target positions
    xinits = torch.as_tensor(np.tile(np.array(plant.x_init, dtype=np.float32), (B, 1)))
    ang = npr.rand(B, 1) * 2 * np.pi if task.rand_tar else np.ones((B, 1)) * np.pi / 2
    xtargs = torch.as_tensor(np.concatenate((np.cos(ang), np.sin(ang) + 1), axis=-1, dtype=np.float32))

    # go cue
    starts = np.round((npr.rand(B) * (task.g_lims[1] - task.g_lims[0]) + task.g_lims[0]) / dt).astype(int)
    stops = starts + int(task.move_time / dt)
    g = torch.as_tensor(np.stack(
        [np.concatenate((np.ones((starts[bi], 1)), np.zeros((NT - starts[bi], 1))), dtype=np.float32) for bi in
         range(B)], axis=1))

    # random initialization of hidden state
    h = torch.as_tensor(net.sig * npr.randn(B, N).astype(np.float32))

    # initial arm pos and velocity in angular coordinates
    w = torch.as_tensor(np.tile(plant.w_init, (B, 1)), dtype=torch.float32)
    v = torch.zeros(B, R)

    # initial noise
    noise = torch.as_tensor(npr.randn(B, R).astype(np.float32)) * plant.noise_scale

    # save tensors for plotting
    ha = torch.zeros(NT, B, N)  # save the hidden states for each time bin for plotting
    ua = torch.zeros(NT, B, R)  # angular acceleration of joints
    na = torch.as_tensor(npr.randn(NT, B, R).astype(np.float32)) * plant.noise_scale  # multiplicative noise
    xa = torch.zeros(NT, B, R)  # angular position of joints

    for ti in range(NT):
        # network update
        s = torch.cat((xtargs, g[ti]), dim=-1)  # input
        h = h + dt * (-h + torch.tanh(s.mm(net.ws) + h.mm(net.J) + net.b))  # dynamics
        u = h.mm(net.wr)  # output

        # physics
        noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
        v, w, x = plant.forward(u, v, w, noise, dt)     # actual state

        # save values for plotting
        ha[ti], ua[ti], na[ti], xa[ti] = h, u, noise, x

    # loss is sum of mse and regularization terms
    mse, u_reg, du_reg = task.loss(dt, xa, ua, xinits, xtargs, starts, stops, algo.lambda_u, algo.lambda_du)
    loss = mse + u_reg + du_reg

    # save loss, mse, and regularization terms
    losses.append(loss.item())
    mses.append(mse.item())
    u_regs.append(u_reg.item())
    du_regs.append(du_reg.item())

    # print loss
    print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(loss.item()), end='')

    # do BPTT
    loss.backward()
    opt.step()
    opt.zero_grad()

    # plot
    if doplot and ei == 0:
        plot(num_to_plot, ei, dt, t, task.T, ha, ua, na, noise, xa, xinits, xtargs, starts, stops, mses, u_regs, du_regs,
             fig=fig, with_noise=True)
    elif doplot and ei % plot_every == 0:
        plot(num_to_plot, ei, dt, t, task.T, ha, ua, na, noise, xa, xinits, xtargs, starts, stops, mses, u_regs, du_regs,
             init=False, with_noise=True)
    elif doplot and ei == (algo.Nepochs-1):
        # print training time
        print("; train time: ", time() - prevt)
        plot(num_to_plot, ei, dt, t, task.T, ha, ua, na, noise, xa, xinits, xtargs, starts, stops, mses, u_regs, du_regs,
             init=False, with_noise=True)

## save results
torch.save({'net': net, 'task': task, 'plant': plant, 'algo': algo, 'mses': mses}, 'bptt__reaching.pt')
