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

# +
import sys

from nflib.flows import (
    AffineConstantFlow,
    ActNorm,
    Invertible1x1Conv,
    NormalizingFlow,
    NormalizingFlowModel,
)
from nflib.spline_flows import NSF_CL
# -

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print('Using the cpu...')

# !ls

# +
# choose data here
data = np.loadtxt("x_moons.csv")
print(data.shape)

data = torch.Tensor(data)
dim = data.shape[-1]
print(dim)
# -

# choose prior here
base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

# configure the normalising flow
nfs_flow = NSF_CL
#flows = [nfs_flow(dim=dim, K=8, B=3, hidden_dim=128) for _ in range(3)]
convs = [Invertible1x1Conv(dim=dim) for _ in range(3)]
norms = [ActNorm(dim=dim) for _ in range(3)]
flows = list(itertools.chain(*zip(norms, convs)))

# initialise the model
model = NormalizingFlowModel(prior, flows)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-6, weight_decay=0)  # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

# train_loader
train_loader = torch.utils.data.DataLoader(
    data, batch_size=128, shuffle=True, pin_memory=True)

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
        loss = -torch.mean(logprob).to(device)  # NLL

        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu())

    if k % 100 == 0:
        print("Loss at step k =", str(k) + ":", loss.item())

t1 = time()
print(f'Elapsed time: {t1-t0:.1f} s')

elapsed_time = [t1-t0]

path = f"model_moonies_ngpus_test.pth"
torch.save(model.state_dict(), path)
np.savetxt('moonies_loss.npy', loss_history)
np.savetxt('apogee_batch_elapsed_time.npy', elapsed_time)

print("Hooray. You're done.")
print("Saved model to:", path)


