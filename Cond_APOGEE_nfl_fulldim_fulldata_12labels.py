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

# +
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print('Using the cpu...')

# +
spectra = np.loadtxt('./data/apogee_batch_Xtrain_full.csv')[:, :1000]
spectra = spectra.T

print(spectra.shape)
# -

#use even number of dimensions
spectra = spectra[:, ::3]
spectra = torch.Tensor(spectra)
spectra = spectra - 0.5
dim = spectra.shape[-1]
print(dim)

labels = np.loadtxt("./data/apogee_batch_ytrain_full.csv")[:1000]
labels.shape

y = np.array([labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4], labels[:, 5], \
              labels[:, 9], labels[:, 10], labels[:, 11], labels[:, 13], labels[:, 15], labels[:, 18]]).T

y[:, 0] = y[:, 0]/1000

x = spectra
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 12)
print(y.shape)

print('x shape', x.shape) #choose all the conditions
print('y shape', y.shape) #choose the first three labels to condition on

# choose prior here
dim = x.shape[-1]
print('x dim is:', dim)

cond_dim = y.shape[-1]
print('y dim is:', cond_dim)

# configure the normalising flow

n_units = 50

nfs_flow = NSF_CL
flows = [nfs_flow(dim=dim, device=device, context_features=cond_dim, K=8, B=3, hidden_dim=128) for _ in range(n_units)]
convs = [Invertible1x1Conv(dim=dim, device=device) for _ in flows]
norms = [ActNorm(dim=dim, device=device) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

# initialise the model
model = NormalizingFlowModel(prior, flows, device)

# +
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

# optimizer
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-7)  # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))
# -

# train_loader
dataset = TensorDataset(x, y)


# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
loader = DataLoader(dataset, batch_size=50, shuffle=True) #this will give x, y per batch
del dataset
del x
import gc
gc.collect()
torch.cuda.empty_cache()

n_epochs = 500
#loss_history=[]
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
 #       loss_history.append(float(loss))
    if k % 100 == 0:
        print("Loss at step k =", str(k) + ":", loss.item())

# +
# save the loss history too
# add the checkpoint
# multi-GPU
# time the multi gpu
# -

#save checkpoint...
np.savetxt('cond_apogee_fulldim_fulldata_12y.npy', loss_history)
torch.save({'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'cond_apogee_fulldim_fulldata_ngpus_12y.pth')
