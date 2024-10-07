from matplotlib import pyplot as plt
import torch
import numpy as np


def plot(B, epoch, dt, t, T, ha, ua, na, noise, xa, xinits, xtargs, starts, stops, mses, u_regs, du_regs,
          fig=None, init=True, with_noise=False):
    if init:
        plot.fig = plt.figure(figsize=(12, 7)) if fig is None else fig
        plot.a = [plt.subplot(2, 1 + B, ai + 1) for ai in range(2 * (1 + B))]

    # loss plot
    a = plot.a[0]
    a.clear()
    plt.sca(a)
    plt.plot(mses, 'k')
    plt.plot(u_regs, '--', color='tab:red')
    plt.plot(du_regs, '--', color='tab:olive')
    plt.yscale('log')
    plt.legend(['MSE', '$\lambda_1|u|^2$', '$\lambda_2|du/dt|^2$'])
    plt.ylabel('Loss')
    plt.xlabel('Trial')

    # trajectories plot
    a = plot.a[B + 1]
    a.clear()
    plt.sca(a)
    for b in range(B):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][b]
        plt.plot(xa[:, b, 0].detach(), xa[:, b, 1].detach(), color=color)
        plt.scatter(xinits[b, 0], xinits[b, 1], c='k')
        plt.scatter(xtargs[b, 0], xtargs[b, 1], color=color)
    plt.xlim((-1.2, 1.2))
    plt.ylim((-0.2, 2.2))
    plt.title('Trajectory')
    plt.ylabel('x_2 (m)')
    plt.xlabel('x_1 (m)')

    for b in range(B):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][b]

        # network output plot
        a = plot.a[b + 1]
        a.clear()
        plt.sca(a)
        tmp = ua[:, b].detach()
        if with_noise:
            noise = tmp + torch.linalg.norm(tmp, dim=-1, keepdim=True) * na[:, b].detach()
        max_u = (noise if with_noise else tmp).abs().max()
        plt.plot([starts[b] * dt, starts[b] * dt], [-1.05 * max_u, 1.05 * max_u], 'k', linewidth=0.5)
        plt.plot([stops[b] * dt, stops[b] * dt], [-1.05 * max_u, 1.05 * max_u], 'k', linewidth=0.5)
        if with_noise:
            plt.plot(t, noise[:, 0], c=color, linewidth=0.5)
            plt.plot(t, noise[:, 1], c=color, linewidth=0.5, alpha=0.25)
        plt.plot(t, tmp[:, 0], c=color)
        plt.plot(t, tmp[:, 1], c=color, alpha=0.5)
        #         plt.ylim((-0.04, 0.04))
        plt.title('Trial type: autonomous')
        # plt.title('Trial type: autonomous' if np.array_equal([0, 1], c[b]) else 'Trial type: with feedback')
        if b == 0: plt.ylabel('Angular Acc (rad/s^2)')

        # hand position plot
        a = plot.a[B + 2 + b]
        a.clear()
        plt.sca(a)
        plt.plot([starts[b] * dt, starts[b] * dt], [-1, 2], 'k', linewidth=0.5)
        plt.plot([stops[b] * dt, stops[b] * dt], [-1, 2], 'k', linewidth=0.5)
        plt.plot([0, starts[b] * dt], [xinits[b, 0], xinits[b, 0]], 'k', linewidth=2)
        plt.plot([0, starts[b] * dt], [xinits[b, 1], xinits[b, 1]], 'k', linewidth=2)
        plt.plot([stops[b] * dt, T], [xtargs[b, 0], xtargs[b, 0]], 'k', linewidth=2)
        plt.plot([stops[b] * dt, T], [xtargs[b, 1], xtargs[b, 1]], 'k', linewidth=2)
        plt.plot(t, xa[:, b, 0].detach(), c=color)
        plt.plot(t, xa[:, b, 1].detach(), c=color, alpha=0.25)
        plt.ylim((-1.3, 2.3))
        plt.xlabel('Time')
        if b == 0:
            plt.ylabel('Arm Pos (m)')

    plt.suptitle(f'epoch: {epoch}')
    plt.tight_layout()
    plot.fig.canvas.draw()
    plt.show()

def plotdata(mses, conds, trials, uas, xas, dt, B=3):
    NT = np.shape(xas[0])[0]
    t = dt * np.arange(NT)

    plot.fig = plt.figure(figsize=(12, 7))
    plot.a = [plt.subplot(2, 1 + B, ai + 1) for ai in range(2 * (1 + B))]

    # loss plot
    a = plot.a[0]
    a.clear()
    plt.sca(a)
    for idx in np.unique(conds):
        plt.plot([mse for mse, cond in zip(mses, conds) if cond==idx])
    plt.yscale('log'), #plt.xscale('log')
    #plt.legend(['MSE', '$\lambda_1|u|^2$', '$\lambda_2|du/dt|^2$'])
    plt.ylabel('Loss')
    plt.xlabel('Trial')

    # trajectories plot
    a = plot.a[B + 1]
    a.clear()
    plt.sca(a)
    for b in range(B):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][b % 4]
        plt.plot(xas[b][:, 0, 0], xas[b][:, 0, 1], color=color)
        plt.scatter(trials[b].xinits[0, 0], trials[b].xinits[0, 1], c='k')
        plt.scatter(trials[b].xtargs[0, 0], trials[b].xtargs[0, 1], color=color)
    plt.xlim((-2.2, 2.2))
    plt.ylim((-0.2, 2.2))
    plt.title('Trajectory')
    plt.ylabel('x_2 (m)')
    plt.xlabel('x_1 (m)')

    for b in range(B):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][b % 4]

        # network output plot
        a = plot.a[b + 1]
        a.clear()
        plt.sca(a)
        tmp = uas[b][:, 0]
        max_u = np.abs(tmp).max(axis=0)
        plt.plot([trials[b].starts[0] * dt, trials[b].starts[0] * dt], [-1.05 * max_u, 1.05 * max_u], 'k', linewidth=0.5)
 #       plt.plot([trials[b].stops[0] * dt, trials[b].stops[0] * dt], [-1.05 * max_u, 1.05 * max_u], 'k', linewidth=0.5)
        plt.plot(t, tmp[:, 0], c=color)
        plt.plot(t, tmp[:, 1], c=color, alpha=0.5)
        #         plt.ylim((-0.04, 0.04))
        if b == 0: plt.ylabel('Angular Acc (rad/s^2)')

        # hand position plot
        a = plot.a[B + 2 + b]
        a.clear()
        plt.sca(a)
        plt.plot([trials[b].starts[0] * dt, trials[b].starts[0] * dt], [-1, 2], 'k', linewidth=0.5)
#        plt.plot([trials[b].stops[0] * dt, trials[b].stops[0] * dt], [-1, 2], 'k', linewidth=0.5)
        plt.plot([0, trials[b].starts[0] * dt], [trials[b].xinits[0, 0], trials[b].xinits[0, 0]], 'k', linewidth=2)
        plt.plot([0, trials[b].starts[0] * dt], [trials[b].xinits[0, 1], trials[b].xinits[0, 1]], 'k', linewidth=2)
#        plt.plot([trials[b].stops[0] * dt, NT * dt], [trials[b].xtargs[0, 0], trials[b].xtargs[0, 0]], 'k', linewidth=2)
#        plt.plot([trials[b].stops[0] * dt, NT * dt], [trials[b].xtargs[0, 1], trials[b].xtargs[0, 1]], 'k', linewidth=2)
        plt.plot(t, xas[b][:, 0, 0], c=color)
        plt.plot(t, xas[b][:, 0, 1], c=color, alpha=0.25)
        plt.ylim((-1.3, 2.3))
        plt.xlabel('Time')
        if b == 0:
            plt.ylabel('Arm Pos (m)')

    plt.tight_layout()
    plt.show()

