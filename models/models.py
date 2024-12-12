
import torch
import torch.nn as nn
import numpy as np
import pdb


class FeedForward(nn.Module):
    #So bog basic it should tap for black mana
    #DT That one's for you
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 30)
        self.linear2 = nn.Linear(30,30)
        self.linear3 = nn.Linear(30,30)
        self.linear4 = nn.Linear(30,4)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = torch.clamp(x, -1e6, 1e6)
        return x

def derivatives(t, state, g):
    m1, m2, L1, L2 = 1,1,1,1
    if torch.nan in state: pdb.set_trace()
    if torch.inf in state or -torch.inf in state: pdb.set_trace()
    theta1, theta2, z1, z2 = torch.split(state, 1, dim=2)
    delta = theta2 - theta1
    if torch.nan in delta: pdb.set_trace()
    if torch.inf in delta or -torch.inf in delta: pdb.set_trace()

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    denominator2 = (L2 / L1) * denominator1

    dtheta1_dt = z1
    dz1_dt = (
        (m2 * L1 * z1 ** 2 * np.sin(delta) * np.cos(delta)
         + m2 * g * np.sin(theta2) * np.cos(delta)
         + m2 * L2 * z2 ** 2 * np.sin(delta)
         - (m1 + m2) * g * np.sin(theta1))
        / denominator1
    )
    if torch.nan in dz1_dt: pdb.set_trace()
    dtheta2_dt = z2
    dz2_dt = (
        (-m2 * L2 * z2 ** 2 * np.sin(delta) * np.cos(delta)
         + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
         - (m1 + m2) * L1 * z1 ** 2 * np.sin(delta)
         - (m1 + m2) * g * np.sin(theta2))
        / denominator2
    )

    return np.concatenate([dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt], axis=2)

def rk4_derivs(state, f, h, g):
    t = 0
    k1 = f(t, state, g)
    k2 = f(t + h*0.5, state + k1*h*0.5, g)
    k3 = f(t + h*0.5, state + k2*h*0.5, g)
    k4 = f(t + h, state + k3*h, g)
    return (k1 + 2*k2 + 2*k3 + k4) / 6

class PINNLoss(nn.Module):
    def __init__(self, tstep, lamb=.5, g=9.81):
        super().__init__()
        self.tstep = tstep
        self.lamb = lamb
        self.data_crit = nn.MSELoss()
        self.physics_crit = nn.MSELoss()
        self.g = g

    def forward(self, y_pred, y_true, input_data):
        data_loss = self.data_crit(y_pred, y_true)

        empirical_derivs = (y_pred - input_data) / self.tstep
        true_derivs = torch.tensor(rk4_derivs(input_data, derivatives, self.tstep, self.g))
        physics_loss = self.physics_crit(empirical_derivs, true_derivs)

        return self.lamb * data_loss + (1 - self.lamb) * physics_loss




class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.num_calls = 0
        self.net = nn.Sequential(
            nn.Linear(5, 64),  # state: (theta1, w1, theta2, w2, t)
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
        )

    def forward(self, t, y):
        # y has shape (batch_size, 4)
        self.num_calls += 1
        return self.net(y)

