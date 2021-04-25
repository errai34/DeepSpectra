"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

# +
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import sys
sys.path.append('../')

from nf import utils
from nf.utils import torchutils
# -
from nf.nets import MLP
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim).to(self.device))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(
            torch.triu(U, diagonal=1)
        )  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim).to(self.L.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, context):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z, context):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, device, scale=True, shift=True):
        super().__init__()
        self.device = device
        self.s = (
            nn.Parameter(torch.randn(1, dim, requires_grad=True).to(device)) if scale else None
        )
        self.t = (
            nn.Parameter(torch.randn(1, dim, requires_grad=True).to(device)) if shift else None
        )

    def forward(self, x, context):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1).to(self.device)
        return z, log_det

    def backward(self, z, context):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x, context):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x, context)


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, device):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.device = device

    def forward(self, x, context):

        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, context)
            log_det += ld.to(self.device)
            zs.append(x)
        return zs, log_det

    def backward(self, z, context):
        m, _ = z.shape
#        m = z.shape[0]
        log_det = torch.zeros(m).to(self.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, context)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows, device, embedding_net=None):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows, device)

        if embedding_net is not None:
            assert isinstance(
                embedding_net, torch.nn.Module
            ), "embedding_net is not a nn.Module. "
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

        self.device = device

    def forward(self, x, context):

        if context is not None:

            #embedded_context = self._embedding_net(context)
            embedded_context = context
            zs, log_det = self.flow.forward(x, context=embedded_context)
            prior_logprob = self.prior.log_prob(zs[-1].to(self.device))
            prior_logprob = prior_logprob.view(x.size(0), -1).sum(1).to(self.device)


        else:
            zs, log_det = self.flow.forward(x, context=None)
            prior_logprob = self.prior.log_prob(zs[-1].to(self.device))
            prior_logprob = prior_logprob.view(x.size(0), -1).sum(1).to(device)
            #prior_logprob = (
             #  self.prior.log_prob(zs[-1], context=embedded_context)
            #   .view(x.size(0), -1)
            #   .sum(1)
         #  )

        return zs, prior_logprob, log_det

    def backward(self, z, context):

        if context is not None:

            #embedded_context = self._embedding_net(context)
            embedded_context = context
            xs, log_det = self.flow.backward(z, context=embedded_context)

        else:
            xs, log_det = self.flow.backward(z, context=None)

        return xs, log_det

    def sample(self, num_samples, context):

        if context is not None:
            #embedded_context = self._embedding_net(context)
            embedded_context = context
            z = self.prior.sample((num_samples,)) #changed frum num_samples simple
            #z = self.prior.sample(num_samples, context=embedded_context)
            #z = torchutils.merge_leading_dims(z, num_dims=2)
            #embedded_context = torchutils.repeat_rows(
            #    embedded_context, num_reps=num_samples[0]
            #)
            xs, _ = self.flow.backward(z, context=embedded_context)
            #xs = torchutils.split_leading_dim(xs, shape=[-1, num_samples])

        else:
            z = self.prior.sample(num_samples)
            xs, _ = self.flow.backward(z, context=None)

        return xs

#
