import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import (
    Normal,
    MultivariateNormal,
    Uniform,
    TransformedDistribution,
    SigmoidTransform,
)
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer, required
#import pandas as pd

from nf.flows import (
    AffineConstantFlow,
    ActNorm,
    Invertible1x1Conv,
    NormalizingFlow,
    NormalizingFlowModel,
)
from nf.spline_flows import NSF_CL

import itertools
import numpy as np
import matplotlib.pyplot as plt
from time import time

from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print('Using the cpu...')

spectra = np.loadtxt('./data/apogee_batch_Xtrain_full.csv')
spectra = spectra.T

spectra = spectra[::10, :]
print(spectra.shape)

waves = np.loadtxt('waves.npy')
print(waves.shape)

lambda_max = 16200
lambda_min = 15900

m = (waves < lambda_max) & (waves > lambda_min)

new_waves = waves[m]
spectra_new = spectra[:, m]

#use even number of dimensions
spectra_new = spectra_new[:, 1:]
spectra_new = torch.Tensor(spectra_new)
spectra_new = spectra_new - 0.5
dim = spectra_new.shape[-1]
print(dim)

labels = np.loadtxt("./data/apogee_batch_ytrain_full.csv")
labels.shape

labels = labels[::10]

#conditioning on teff, logg, vturb, vmac, ch, nh, oh, cah, ceh, feh
y = np.array([labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4], labels[:, 5], \
              labels[:, 8], labels[:, 10], labels[:, 13],  labels[:, 18]]).T

y[:, 0] = y[:, 0]/1000

x = spectra_new
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 10)
print(y.shape)

print('x shape', x.shape) #choose all the conditions
print('y shape', y.shape) #choose the first three labels to condition on

# choose prior here
dim = x.shape[-1]
print('x dim is:', dim)

cond_dim = y.shape[-1]
print('y dim is:', cond_dim)

n_units = 30

nfs_flow = NSF_CL
flows = [nfs_flow(dim=dim, device=device, context_features=cond_dim, K=8, B=3, hidden_dim=128) for _ in range(n_units)]
convs = [Invertible1x1Conv(dim=dim, device=device) for _ in flows]
norms = [ActNorm(dim=dim, device=device) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

# initialise the model
model = NormalizingFlowModel(prior, flows, device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

# optimizer
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-7)  # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

# train_loader
dataset = TensorDataset(x, y)

# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
loader = DataLoader(dataset, batch_size=100, shuffle=True, pin_memory=True) #this will give x, y per batch

torch.cuda.empty_cache()

n_epochs = 500
loss_history=[]
model.train()
print("Started training")
for k in range(n_epochs):
    for batch_idx, data_batch in enumerate(loader):
        x, y = data_batch
        x = x.to(device)
        y = y.to(device)
        zs, prior_logprob, log_det = model(x, context=y) #definitely need to make this work better!?
        del x
        del y
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob)  # NL

        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss))
    if k % 100 == 0:
        print("Loss at step k =", str(k) + ":", loss.item())

torch.save({'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'cond_apogee_450pixels_2000data_10labels_sparse.pth')

np.savetxt('cond_apogee_450pixels_2000data_10labels_sparse.npy',  loss_history)
