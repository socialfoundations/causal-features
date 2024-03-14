"""Python script to define AND-Mask models.

Created for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""
import copy
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model


class ANDMaskModel(DomainGeneralizationModel):
    """
    Class to represent AND-Mask models of "Learning Explanations that are Hard to Vary" [https://arxiv.org/abs/2009.00329].

    Based on implementation from
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    which is based on https://github.com/gibipara92/learning-explanations-hard-to-vary
    """

    def __init__(self, tau, **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)

        self.tau = tau
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = apply_model(self, x).squeeze()

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0
