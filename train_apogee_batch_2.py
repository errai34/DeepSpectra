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

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif device.type == "cpu":
    print('Using the cpu...')

# +
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
# using 268 dimensions only

plt.plot(spectra[0])

# choose prior here
base_mu, base_cov = torch.zeros(dim).to(device), torch.eye(dim).to(device)
prior = MultivariateNormal(base_mu, base_cov)

# configure the normalising flow
nfs_flow = NSF_CL
flows = [nfs_flow(dim=dim, K=8, B=3, hidden_dim=128) for _ in range(10)]
convs = [Invertible1x1Conv(dim=dim) for _ in flows]
norms = [ActNorm(dim=dim) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# initialise the model
model = NormalizingFlowModel(prior, flows)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)
#spectra = spectra.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-6, weight_decay=1e-5)  # todo tune WD
#print("number of params: ", sum(p.numel() for p in model.parameters()))

#optimizer = RAdam(model.parameters(), lr=5e-10)#
#decayRate = 1e-5
#lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# -

# train_loader
train_loader = torch.utils.data.DataLoader(
    spectra, batch_size=100, shuffle=True, pin_memory=True)

t0 = time()

model.train()
print("Started training")
n_epochs = 10000
loss_history=[]

for k in range(n_epochs):
    for batch_idx, data_batch in enumerate(train_loader):
        x = data_batch.to(device)
        zs, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob)  # NLL

        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu())

    if k % 500 == 0:
        print("Loss at step k =", str(k) + ":", loss.item())

t1 = time()
print(f'Elapsed time: {t1-t0:.1f} s')

elapsed_time = [t1-t0]

# +
path = f"model_apogee_minibatch_1gpus_2.pth"
torch.save(model.state_dict(), path)
np.savetxt('apogee_batch_miniloss_2.npy', loss_history)
#np.savetxt('apogee_batch_elapsed_time.npy', elapsed_time)
# -

print("Hooray. You're done.")
print("Saved model to:", path)

# +
#fname = f"model_apogee_minibatch_1gpus.pth"
#state_dict = torch.load(fname, map_location=torch.device('cpu'))

#model = NormalizingFlowModel(prior, flows).to(device)

#model.eval()
# -

model.eval()
zs = model.sample(50)
z = zs[-1]
z = z.to('cpu')
z = z.detach().numpy()
fig = plt.figure()
for i in range(10):
    plt.plot(z[i])

plt.savefig('test_2.png')
