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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# choose data here
spectra = np.loadtxt("./data/apogee_batch_Xtrain.csv")
spectra = spectra.T
print(spectra.shape)

#use even number of dimensions
spectra = spectra[:, ::9]
spectra = torch.Tensor(spectra)
spectra = spectra - 0.5
dim = spectra.shape[-1]
print(dim)

labels = np.loadtxt("./data/apogee_batch_ytrain.csv")
labels.shape

y = labels[:, :3] # choose teff, log, feh
print(y.shape)

y[:, 0] = y[:, 0]/1000

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

# configure the normalising flow

n_units = 20

nfs_flow = NSF_CL
flows = [nfs_flow(dim=dim, context_features=cond_dim, K=8, B=3, hidden_dim=128) for _ in range(n_units)]
convs = [Invertible1x1Conv(dim=dim) for _ in flows]
norms = [ActNorm(dim=dim) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

model = NormalizingFlowModel(prior, flows).to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-7)  # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

# train_loader
dataset = TensorDataset(x, y)

# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
loader = DataLoader(dataset, batch_size=100, shuffle=True, pin_memory=True) #this will give x, y per batch

n_epochs = 5000

loss_history=[]
model.train()
print("Started training")
for k in range(n_epochs):
    for batch_idx, data_batch in enumerate(loader):
        x, y = data_batch
        x = x.to(device)
        y = y.to(device)
        zs, prior_logprob, log_det = model(x, context=y) #definitely need to make this work better!? 
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob)  # NLL


        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss)

    if k % 100 == 0:
        print("Loss at step k =", str(k) + ":", loss.item())
    
path = f"268_20_apogee.pth"
torch.save(model.state_dict(), path)


