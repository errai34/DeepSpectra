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

# import pandas as pd


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print("Using the cpu...")

# choose data here
spectra = np.load("./data/X_train_payne_region_cond_temp_logg.npy")
spectra = spectra.T
print(spectra.shape)

# use even number of dimensions
spectra = spectra[:, 1:]
spectra = torch.Tensor(spectra)
spectra = spectra - 0.5
dim = spectra.shape[-1]
print('spectra dim is', dim)
print(spectra.shape)

# conditional labels here

labels = np.load("./data/y_train_payne_region_cond_temp_logg.npy")
labels.shape

# conditioning on teff, logg
y = np.array([labels[:, 0], labels[:, 1]]).T
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 2)
print(y.shape)

cond_dim = y.shape[-1]
print("y dim is:", cond_dim)

# choose prior here
base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

# configure the normalising flow
nfs_flow = NSF_CL
flows = [
    nfs_flow(
        dim=dim, device=device, context_features=cond_dim, K=8, B=3, hidden_dim=256
    )
    for _ in range(30)
]  # things to change> maybe more is needed??!
convs = [Invertible1x1Conv(dim=dim, device=device) for _ in flows]
norms = [ActNorm(dim=dim, device=device) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# initialise the model
model = NormalizingFlowModel(prior, flows, device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-6, weight_decay=1e-5)  # todo tune WD
# print("number of params: ", sum(p.numel() for p in model.parameters()))

# train_loader
dataset = TensorDataset(spectra, y)

# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
loader = DataLoader(
    dataset, batch_size=100, shuffle=True, pin_memory=True
)  # this will give x, y per batch

t0 = time()
model.train()
print("Started training")
n_epochs = 500
loss_history = []


# loss_history=[]
model.train()
print("Started training")
for k in range(n_epochs):
    for batch_idx, data_batch in enumerate(loader):
        x, y = data_batch
        x = x.to(device)
        y = y.to(device)
        zs, prior_logprob, log_det = model(
            x, context=y
        )  # definitely need to make this work better!?
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

t1 = time()
print(f"Elapsed time: {t1-t0:.1f} s")

# Specify a path to save to
PATH = "model_cond_exp1.pt"
# Save
torch.save(model.module.state_dict(), PATH)
np.savetxt("loss_hist_cond_exp1.npy", loss_history)

model.eval()
cont = np.ones((100, 2))
cont[:, 0] = 4.5
cont[:, 1] = 2.1

cont = torch.tensor(cont, dtype=torch.float32).reshape(-1, 2)

zs = model.sample(100, context=cont)
z = zs[-1]
z = z.to('cpu')
z = z.detach().numpy()

fig = plt.figure(figsize=(14, 4))

for i in range(10):
    plt.plot(z[i])

plt.savefig('model_cond_exp1.png')
