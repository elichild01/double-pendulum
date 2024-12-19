from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from neural_ode import ODEFunc, Pendulum_Data
import torch
from torchdiffeq import odeint
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(num_epochs, dims, std_dev, min_steps):
    max_steps = min_steps + 5
    G = 9.81
    delta_t = 0.005
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 1
    lam = 0.5
    save_every = 100
    val_every = 25
    val_size = 100

    # Set model, optimizer and loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ODEFunc(dims)
    criterion = nn.MSELoss()
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    # Training
    data = Pendulum_Data(min_steps, max_steps, G, delta_t, batch_size)
    dl = DataLoader(data, batch_size=batch_size)
    train_losses = []
    val_losses = []

    for i in tqdm(range(num_epochs)):
        train_losses_local = []
        # each batch is as long as the dataset, so an epoch is one batch
        for X_batch, y_batch in dl:
            X_batch += np.random.normal(0, std_dev, X_batch.shape)
            X_batch, y_batch = X_batch.to(torch.float), y_batch.to(torch.float)
            optimizer.zero_grad()

            # get prediction and make a tuple to pass to the loss
            y_pred = odeint(model, X_batch[:, :, 4:], delta_t * torch.arange(len(X_batch)+1))[1:]
            y_batch = y_batch[:, :, 4:]  # remove mass/length information from outputs
            loss_args = (y_pred.squeeze(), y_batch.squeeze())

            # calculate loss and backprop
            loss = criterion(*loss_args)
            loss.backward()
            train_losses_local.append(loss.item())
            optimizer.step()
        train_losses.append(np.mean(train_losses_local))

        if (i + 1) % val_every == 0:
            model.eval()
            val_losses_local = []

            # run val_size batches to test validation error
            for j in range(val_size):
                for X_batch, y_batch in dl:
                    X_batch += np.random.normal(0, std_dev, X_batch.shape)
                    X_batch, y_batch = X_batch.to(torch.float), y_batch.to(torch.float)
                    optimizer.zero_grad()

                    # get prediction
                    y_pred = None
                    loss_args = None
                    y_pred = odeint(model, X_batch[:, :, 4:], delta_t * torch.arange(len(X_batch) + 1))[1:]
                    y_batch = y_batch[:, :, 4:]  # remove mass/length information from outputs
                    loss_args = (y_pred.squeeze(), y_batch.squeeze())

                    # calculate loss, then average
                    loss = criterion(*loss_args)
                    val_losses_local.append(loss.item())
            val_losses.append(np.mean(val_losses_local))
            model.train()

    return model, train_losses, val_losses

def plot_results(train_losses, val_losses, ose):
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.plot(train_losses)
    plt.title("Training Loss")

    plt.subplot(132)
    plt.plot(val_losses)
    plt.title("Validation Loss")

    plt.subplot(133)
    plt.plot(ose)
    plt.title("One Step Error")

    plt.tight_layout()
    plt.show()