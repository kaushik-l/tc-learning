import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

# implement Random Feedback Local Online learning rule (Murray 2019)

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
S, N, R, g, dt = 1, 50, 1, 1.5, .1
ws = (2 * npr.random((N, S)) - 1) / np.sqrt(S)      # input weights
J = g * npr.standard_normal([N, N]) / np.sqrt(N)    # recurrent weights
wr = (2 * npr.random((R, N)) - 1) / np.sqrt(N)      # readout weights
B = npr.standard_normal([N, R]) / np.sqrt(R)        # feedback weights
z0 = npr.randn(N, 1)                                # initial activity (membrane potential)
h0 = f(z0)                                          # initial activity (firing rate)

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
lr, Nepochs = 0.05, 10000

# track variables during learning
learning = {'epoch': [], 'mses': [], 'fbalignment': []}

# save tensors for plotting
sa = np.zeros((NT, S))  # input
ha = np.zeros((NT, N))  # activity
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
    z, h = z0, h0

    # error
    err = np.zeros((NT, R))

    # eligibility trace
    q = np.zeros((N, N))

    for ti in range(NT):

        # update membrane potential
        Iin = np.matmul(ws, s[:, ti])[:, None]
        Irec = np.matmul(J, h)
        z = Iin + Irec

        # update eligibility trace
        q = dt * df(z) * h.T + (1 - dt) * q

        # update output and error
        h = (1 - dt) * h + dt * f(z)    # activity
        u = np.matmul(wr, h)            # output
        err[ti] = ustar[ti] - u         # error

        # update weights
        wr += (((lr / NT) * h) * err[ti]).T
        J += ((lr / NT) * np.matmul(B, err[ti]).reshape(N, 1) * q)

        # save values for plotting
        ha[ti], ua[ti], = h.T, u.T

    # print loss and update plot
    mse = loss(err)
    if ei % 100 == 0:
        print('\r' + str(ei) + '/' + str(Nepochs) + '\t Err:' + str(mse), end='')
        line2.set_ydata(ua)
        fig.canvas.draw()
        plt.pause(0.0001)
        plt.title('Epoch ' + str(ei) + '/' + str(Nepochs))

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
plt.pause(5)
