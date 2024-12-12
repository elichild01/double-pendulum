
import sys
sys.path.append(sys.path[0]+"\\\\..")  # assuming the first element of sys.path is the path to the scripts folder, this allows imports from within double-pendulum

import numpy as np
from utils.dataset import Pendulum_Data
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.utils.data.dataloader import DataLoader
from models.models import ODEFunc
from models.models import FeedForward
from models.models import PINNLoss
import argparse
from tqdm import tqdm



# Get command line arguments for the filepath, model type, and various params
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--model', type=str, choices=['PINN', 'NODE'], default='NODE')
parser.add_argument('--min_steps', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=1)
parser.add_argument('--G', type=float, default=9.81)
parser.add_argument('--delta_t', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--val_every', type=int, default=25)
parser.add_argument('--val_size', type=int, default=100)
args = parser.parse_args()


# Set model, optimizer and loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
criterion = None
if args.model == 'PINN':
    model = FeedForward()
    criterion = PINNLoss(args.delta_t, args.lam, args.G)
else:
    model = ODEFunc()
    criterion = nn.MSELoss()
model = model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
losses = []

# Training
data = Pendulum_Data(args.min_steps, args.max_steps, args.G, args.delta_t, args.batch_size)
dl = DataLoader(data, batch_size=args.batch_size)
train_losses = []
val_losses = []

for i in tqdm(range(args.num_epochs)):
    train_losses_local = []
    # each batch is as long as the dataset, so an epoch is one batch
    for X_batch, y_batch in dl:
        X_batch, y_batch = X_batch.to(torch.float), y_batch.to(torch.float)
        optimizer.zero_grad()

        # get prediction and make a tuple to pass to the loss
        y_pred = None
        loss_args = None
        if args.model == 'PINN':
            # the last element of X_batch is time, and should not be included
            y_pred = model(X_batch[:, :, :-1])
            loss_args = (y_pred, y_batch[:, :, 4:-1], X_batch[:, :, 4:-1])
        else:
            y_pred = odeint(model, X_batch[:, :, 4:], args.delta_t * torch.arange(len(X_batch)+1))[1:]
            y_batch = y_batch[:, :, 4:]  # remove mass/length information from outputs
            loss_args = (y_pred.squeeze(), y_batch.squeeze())

        # calculate loss and backprop
        loss = criterion(*loss_args)
        loss.backward()
        train_losses_local.append(loss.item())
        optimizer.step()
    train_losses.append(np.mean(train_losses_local))

    if (i + 1) % args.val_every == 0:
        model.eval()
        val_losses_local = []

        # run val_size batches to test validation error
        for j in range(args.val_size):
            for X_batch, y_batch in dl:
                X_batch, y_batch = X_batch.to(torch.float), y_batch.to(torch.float)
                optimizer.zero_grad()

                # get prediction
                y_pred = None
                loss_args = None
                if args.model == 'PINN':
                    # the last element of X_batch is time, and should not be included
                    y_pred = model(X_batch[:, :, :-1])
                    loss_args = (y_pred, y_batch[:, :, 4:-1], X_batch[:, :, 4:-1])
                else:
                    y_pred = odeint(model, X_batch[:, :, 4:], args.delta_t * torch.arange(len(X_batch) + 1))[1:]
                    y_batch = y_batch[:, :, 4:]  # remove mass/length information from outputs
                    loss_args = (y_pred.squeeze(), y_batch.squeeze())

                # calculate loss, then average
                loss = criterion(*loss_args)
                val_losses_local.append(loss.item())
        val_losses.append(np.mean(val_losses_local))
        model.train()

    if (i + 1) % args.save_every == 0:
        torch.save(model.state_dict(), f'{args.path}_{i}.pt')
        torch.save(train_losses, f'{args.path}_{i}_train_losses.pt')
        torch.save(val_losses, f'{args.path}_{i}_val_losses.pt')



