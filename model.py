import numpy as np
import math
import torch
import numpy.random as npr

# seed
npr.seed(4)


class Network:
    def __init__(self, name='Ctx', N=200):
        self.name = name
        # network parameters
        self.N = N  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g_in = 1.0  # initial input weight scale
        self.g_rec = 1.5  # initial recurrent weight scale
        self.g_out = 0.01  # initial output weight scale
        self.S_targ, self.S_go = 2, 1
        self.S = self.S_targ + self.S_go  # input, used here
        self.R = 2  # readout
        self.sig = 0.01  # initial activity scale
        ws0 = self.g_in * np.random.standard_normal([self.S, self.N]).astype(np.float32) / np.sqrt(self.S)
        J0 = self.g_rec * np.random.standard_normal([self.N, self.N]).astype(np.float32) / np.sqrt(self.N)
        wr0 = self.g_out * np.random.standard_normal([self.N, self.R]).astype(np.float32) / np.sqrt(self.N)
        b0 = np.zeros([1, self.N]).astype(np.float32)
        self.ws = torch.tensor(ws0, requires_grad=True)
        self.J = torch.tensor(J0, requires_grad=True)
        self.wr = torch.tensor(wr0, requires_grad=True)
        self.b = torch.tensor(b0, requires_grad=True)

    # add input channels
    def add_inputs(self, s):
        self.S += s

    # add input channels
    def add_units(self, n):
        self.N += n

    # add input channels
    def add_outputs(self, r):
        self.R += r


class Task:
    def __init__(self, name='Reaching', rand_tar=True, num_tar=None, auto=False, feed=False, both=False):
        self.name = name
        # task parameters
        self.T = 35  # duration (in units of tau)
        self.g_lims = (5, 20)       # (tmin, tmax) of go cue
        self.rand_tar = rand_tar    # randomize location?
        self.num_tar = num_tar      # number of target locations (if rand_tar=False)
        self.move_time = 10         # time after go cue to complete the reach

    def loss(self, dt, xt, ut, xinits, xtargs, starts, stops, lambda_u=0, lambda_du=0):
        if self.name == 'Reaching':
            mse = torch.stack([((xt[:start] - xinit) ** 2).mean() + ((xt[stop:] - xtarg) ** 2).mean()
                               for xt, xinit, xtarg, start, stop in zip(xt.transpose(0, 1), xinits, xtargs, starts, stops)
                               ]).mean() / 2
            u_reg = lambda_u * (ut ** 2).sum(dim=-1).mean()
            du_reg = lambda_du * ((torch.diff(ut, dim=0) / dt) ** 2).sum(dim=-1).mean()
            return mse, u_reg, du_reg


class Plant:
    def __init__(self, name='TwoLink'):
        self.name = name
        # physics parameters
        self.noise_scale = 0.05  # network output multiplicative noise scale (0.2)
        self.noise_corr_time = 4  # noise correlation time (units of tau)
        self.drag_coeff = 1.0   # viscosity
        self.w_init = [math.pi / 6, 2 * math.pi / 3]    # initial angles of links
        self.x_init = [0, 1.0]  # initial position of endpoint

    # plant dynamics (actual dynamics with noise)
    def forward(self, u, v, w, noise, dt):
        x = None
        # physics
        if self.name == 'TwoLink':
            accel = u + torch.linalg.norm(u, dim=-1, keepdim=True) * noise - self.drag_coeff * v
            v_new = v + accel * dt
            w = w + v * dt + 0.5 * accel * dt ** 2
            v = v_new
            # hand location
            ang1, ang2 = w[:, 0], w.sum(dim=-1)
            x = torch.stack((torch.cos(ang1) + torch.cos(ang2), torch.sin(ang1) + torch.sin(ang2)), dim=-1)
        # return
        return v, w, x

    # predicted plant dynamics (predicted dynamics unaware of noise)
    def forward_predict(self, u, v, w, dt):
        x_predict = None
        # physics
        if self.name == 'TwoLink':
            accel = u - self.drag_coeff * v
            v_new = v + accel * dt
            w = w + v * dt + 0.5 * accel * dt ** 2
            v = v_new
            # hand location
            ang1, ang2 = w[:, 0], w.sum(dim=-1)
            x_predict = torch.stack((torch.cos(ang1) + torch.cos(ang2), torch.sin(ang1) + torch.sin(ang2)), dim=-1)
        # return
        return x_predict


class Algorithm:
    def __init__(self, name='Adam', Nepochs=1000, B=40, lr=1e-4, online=False):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 100000
        self.B = B  # batches per epoch
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-6
        self.lambda_u = 5e-2
        self.lambda_du = 5e-1
        self.online = online


class Trial:
    def __init__(self):
        # learning parameters
        self.xinits = []    # initial states
        self.xtargs = []    # target states
        self.c = []     # context cue
        self.starts = []
        self.stops = []
        self.g = []     # go cue

    def add_xinits(self, plant, B=1):
        self.xinits = torch.as_tensor(np.tile(np.array(plant.x_init, dtype=np.float32), (B, 1)))

    def add_xtargs(self, task, loc=0, B=1):
        if task.rand_tar:
            ang = npr.rand(B, 1) * 2 * np.pi
        else:
            angs = 2 * np.pi * np.arange(1, task.num_tar+1)/task.num_tar + np.pi / 2
            ang = np.ones((B, 1)) * angs[loc]
        self.xtargs = torch.as_tensor(np.concatenate((np.cos(ang), np.sin(ang) + 1), axis=-1, dtype=np.float32))

    def add_go(self, task, dt, B=1):
        NT = int(task.T / dt)
        self.starts = np.round((npr.rand(B) * (task.g_lims[1] - task.g_lims[0]) + task.g_lims[0]) / dt).astype(int)
        self.stops = self.starts + int(task.move_time / dt)
        self.g = torch.as_tensor(np.stack(
            [np.concatenate((np.ones((self.starts[bi], 1)), np.zeros((NT - self.starts[bi], 1))), dtype=np.float32)
             for bi in range(B)], axis=1))
