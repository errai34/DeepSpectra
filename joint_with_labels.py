import sys
sys.path.append('./')

import itertools
import numpy as np
import matplotlib.pyplot as plt
from time import time
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

from nflib.flows import (
    AffineConstantFlow,
    ActNorm,
    Invertible1x1Conv,
    NormalizingFlow,
    NormalizingFlowModel,
)
from nflib.spline_flows import NSF_CL

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print('Using the cpu...')

# choose data here
# choose data here
spectra = np.load("./data/X_train_payne_region_uncond.npy")
#spectra = np.load("/content/drive/My Drive/DeepSpectra/DeepSpectra/data/X_train_payne_region_cond_temp_logg.npy")
spectra = spectra.T
print(spectra.shape)

# use even number of dimensions
spectra = spectra[:, :801]
spectra = torch.Tensor(spectra)
spectra = spectra - 0.5
dim = spectra.shape[-1]
print('spectra dim is', dim)
print(spectra.shape)
labels = np.load("./data/y_train_payne_region_uncond.npy")
print('labels shape', labels.shape)

# conditioning on teff, logg
y = np.array([labels[:, 0], labels[:, 1], labels[:, 18]]).T
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 3)
print(y.shape)

cond_dim = y.shape[-1]
print("y dim is:", cond_dim)

new_spectra = torch.cat((spectra, y), 1)
print(new_spectra.size())

dim = new_spectra.shape[-1]
print('New dim after adding the labels is', dim)

# choose prior here
base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

# configure the normalising flow
nfs_flow = NSF_CL
nflows = 20
hidden_dim = 600

flows = [nfs_flow(dim=dim, device=device, K=8, B=3, hidden_dim=hidden_dim) for _ in range(nflows)] #things to change> maybe more is needed??!
convs = [Invertible1x1Conv(dim=dim, device=device) for _ in flows]
norms = [ActNorm(dim=dim, device=device) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# initialise the model
model = NormalizingFlowModel(prior, flows, device=device)

#if torch.cuda.device_count() >1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model)


model = nn.DataParallel(model) #assume this is de facto
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-6, weight_decay=1e-5)  # todo tune WD
#print("number of params: ", sum(p.numel() for p in model.parameters()))

# Training run
#-------------------------------------------------------------------------------------
# train_loader
train_loader = torch.utils.data.DataLoader(
    new_spectra, batch_size=200, shuffle=True, pin_memory=True)

t0 = time()

model.train()
print("Started training")
n_epochs = 500
loss_history=[]

for k in range(n_epochs):
    for batch_idx, data_batch in enumerate(train_loader):
        x = data_batch.to(device)
        zs, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(logprob)  

        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu())

    if k % 100 == 0:
        print("Loss at step k =", str(k) + ":", loss.item())

t1 = time()
print(f'Elapsed time: {t1-t0:.1f} s')

#model = NormalizingFlowModel(prior, flows, device=device)
PATH = "./joint_with_labels_exp2.pt"
# Save
torch.save(model.module.state_dict(), PATH)


zs = model.module.sample(10000)
z = zs[-1]
z = z.to('cpu')
z = z.detach().numpy()

np.save('./samples.npy', z, allow_pickle=True)
