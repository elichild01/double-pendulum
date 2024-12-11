
import argparse
import torch
from models import ODEFunc
from models import FeedForward
import metrics
from dataset import Pendulum_Data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchdiffeq import odeint
import numpy as np


# Get command line arguments for the filepath and model type
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('save_path', type=str)
parser.add_argument('--model', type=str, choices=['PINN', 'NODE'], default='NODE')
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--G', type=float, default=9.81)
parser.add_argument('--delta_t', type=float, default=0.005)
parser.add_argument('--val_size', type=int, default=100)
args = parser.parse_args()


# Set model and dataloader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
if args.model == 'PINN':
    model = FeedForward()
else:
    model = ODEFunc()
model = model.to(device=device)
model.eval()

data = Pendulum_Data(args.num_steps, args.num_steps, args.G, args.delta_t, 1)
dl = DataLoader(data, batch_size=1, shuffle=False)


# evaluate
oses = []
ttds = []
tes = []
ges = []
for i in tqdm(range(args.val_size)):
    for X_batch, y_batch in dl:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # get prediction
        y_pred = None
        if args.model == 'PINN':
            y_pred = model(X_batch)
        else:
            y_pred = odeint(model, X_batch[0, :, 4:], args.delta_t * np.arange(len(X_batch) + 1))[1:]
            y_batch = y_batch[0, :, 4:]  # remove mass/length information from outputs

        y_pred = y_pred.cpu().detach().numpy().squeeze()
        y_batch = y_batch.cpu().detach().numpy().squeeze()

    # calculate metrics
    ose = metrics.one_step_error(y_pred, y_batch, lambda_=1)
    ttd = metrics.time_to_divergence(y_pred, y_batch, lambda_=1) * args.delta_t
    te = [metrics.total_divergence_at_time(idx, y_pred, y_batch, lambda_=1) for idx in range(len(y_pred))]
    ge = [metrics.global_error(idx, y_pred, y_batch, lambda_=1) for idx in range(1,len(y_pred))]

    oses.append(ose)
    ttds.append(ttd)
    tes.append(te)
    ges.append(ge)

# save them
torch.save(oses, f'{args.save_path}_oses.pt')
torch.save(ttds, f'{args.save_path}_ttds.pt')
torch.save(tes, f'{args.save_path}_tes.pt')
torch.save(ges, f'{args.save_path}_ges.pt')
