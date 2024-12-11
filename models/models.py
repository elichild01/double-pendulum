
import torch
import torch.nn as nn


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


class PINNLoss(nn.Module):
    def __init__(self, tstep, lamb=.5):
        super().__init__()
        self.tstep = tstep
        self.lamb = lamb
        self.data_crit = nn.MSELoss()
        self.physics_crit = nn.MSELoss()

    def forward(self, y_pred, y_true, input_data):
        data_loss = self.data_crit(y_pred, y_true)

        empirical_derivs = (y_pred - input_data[:, 4:]) / self.tstep
        true_derivs = torch.tensor(rk4_derivs(input_data, derivatives, self.tstep))[:, 4:]
        physics_loss = self.physics_crit(empirical_derivs, true_derivs)

        return self.lamb * data_loss + (1 - self.lamb) * physics_loss




class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.num_calls = 0
        self.net = nn.Sequential(
            nn.Linear(4, 64),  # state: (theta1, w1, theta2, w2)
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
        )

    def forward(self, t, y):
        # y has shape (batch_size, 4)
        self.num_calls += 1
        return self.net(y)

