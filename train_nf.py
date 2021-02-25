import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
)
from nflib.spline_flows import NSF_AR, NSF_CL

#choose data here
spectra = np.loadtxt('./data/spectra.csv')
spectra = torch.Tensor(spectra)
dim = spectra.shape[-1]

#choose prior here
base_mu, base_cov = torch.zeros(dim), torch.eye(dim)
prior = MultivariateNormal(base_mu, base_cov)

#configure the normalising flow
nfs_flow = NSF_CL
flows = [nfs_flow(dim=dim, K=8, B=3, hidden_dim=128) for _ in range(5)]
convs = [Invertible1x1Conv(dim=dim) for _ in flows]
norms = [ActNorm(dim=dim) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# initialise the model
model = NormalizingFlowModel(prior, flows)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=0) # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

#train_loader
train_loader = torch.utils.data.DataLoader(spectra, batch_size=200,\
                                                shuffle=True, pin_memory=True)

model.train()
print("Started training")
for k in range(1000):
    for batch_idx, data_batch in enumerate(train_loader):
        x = data_batch
        zs, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob) # NLL

        model.zero_grad()
        loss.backward()
        optimizer.step()

    if k % 100 == 0:
        print("Loss at step k =", str(k)+":", loss.item())


path = f'../flow_results/model.pth'
torch.save(model.state_dict(), path)

print("Hooray. You're done.")
print("Saved model to:", path)
