import itertools
import numpy as np
import matplotlib.pyplot as plt


import itertools

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from torch.distributions import (
    Normal,
    MultivariateNormal,
    Uniform,
    TransformedDistribution,
    SigmoidTransform,
)

from torch.utils.data import DataLoader, TensorDataset

import flow_torch_conditional as flow_cond

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print('Using the cpu...')


# choose data here
spectra = np.loadtxt("./data/apogee_batch_Xtrain.csv")
spectra = spectra.T
print(spectra.shape)

#use even number of dimensions, in this case 86 dimensions for pixels
spectra = spectra[:, ::9]
spectra = torch.Tensor(spectra)
spectra = spectra - 0.5
dim = spectra.shape[-1]
print(dim)

labels = np.loadtxt("./data/apogee_batch_ytrain.csv")
labels.shape

y = labels[:, :3] # choose teff, log, feh
print(y.shape)

x = spectra
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 3)
print(y.shape)

print('x shape', x.shape) #choose all the conditions
print('y shape', y.shape) #choose the first three labels to condition on

# choose prior here
dim = x.shape[-1]
print('x dim is:', dim)

cond_dim = y.shape[-1]
print('y dim is:', cond_dim)

n_flows = 1

#n_flows = 960 # the directory plummer_flow already has an ensemble of 960 trained flows,
               # skip the following cell if you do not wish to retrain the normalizing flows.

n_dim = dim
n_cond_dim = cond_dim
n_units = 20

n_epochs = 500
batch_size = 100
n_samples = x.shape[0]

n_steps = n_samples * n_epochs // batch_size
print(f'n_steps = {n_steps}')

flow = flow_cond.NormalizingFlow(n_dim, n_cond_dim, n_units)

#optimizer


# optimizer
opt = optim.Adam(flow.parameters(), lr=5e-7)  # todo tune WD

flow = flow.to(device)
# train_loader
dataset = TensorDataset(x, y)

# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True) #this will give x, y per batch

#----------------------------------------------------------------------------------
loss_history = []
i = 0

for e in range(int(n_epochs)):
    for batch_idx, data_batch in enumerate(loader):
        x, y = data_batch
        x = x.to(device)
        y = y.to(device)

        zs, prior_logprob, log_det = flow(x, y)
        logprob = prior_logprob + log_det
        loss = -torch.mean(logprob)

        flow.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())

torch.save(flow.state_dict(), 'mini_apogee_cond_268dim.pth')
