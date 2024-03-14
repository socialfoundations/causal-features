"""Python script to define Causal IRL models.

Created for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""
import copy
import torch
import torch.nn.functional as F
import numpy as np

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model


class AbstractCausIRL(DomainGeneralizationModel):
    """Abstract class for causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646).

    Based on implementation from
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """

    def __init__(self, mmd_gamma, gaussian, **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)

        self.gaussian = gaussian
        self.mmd_gamma = mmd_gamma
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [xi for xi, _ in minibatches]
        classifs = [apply_model(self, fi).squeeze() for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.mmd_gamma*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    """Causality based invariant representation learning algorithm using the MMD distance."""

    def __init__(self, mmd_gamma, **hparams):
        super(CausIRL_MMD, self).__init__(mmd_gamma=mmd_gamma, gaussian=True, **hparams)


class CausIRL_CORAL(AbstractCausIRL):
    """Causality based invariant representation learning algorithm using the CORAL distance."""

    def __init__(self, mmd_gamma, **hparams):
        super(CausIRL_CORAL, self).__init__(mmd_gamma=mmd_gamma, gaussian=False, **hparams)
