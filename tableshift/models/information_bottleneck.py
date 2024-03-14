"""Python script to define Information Bottelneck models.

Created for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""
import copy

import torch
import torch.nn.functional as F
import torch.autograd as autograd

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model


class IBERMModel(DomainGeneralizationModel):
    """Class to represent Information Bottleneck based ERM on feature with conditioning.

    Based on implementation from
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """

    def __init__(self, ib_lambda, ib_penalty_anneal_iters, **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)

        self.ib_lambda = ib_lambda
        self.ib_penalty_anneal_iters = ib_penalty_anneal_iters

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.ib_lambda if self.update_count
                             >= self.ib_lambda else
                             0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = apply_model(self, all_x).squeeze()
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_x[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.ib_lambda:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self._init_optimizer()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}


class IBIRMModel(DomainGeneralizationModel):
    """Class to represent Information Bottleneck based IRM on feature with conditionning.

    Based on implementation from
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """

    def __init__(self, ib_lambda, ib_penalty_anneal_iters, irm_lamda, irm_penalty_anneal_iters, **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)

        self.ib_lambda = ib_lambda
        self.ib_penalty_anneal_iters = ib_penalty_anneal_iters
        self.irm_lambda = irm_lamda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters

        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits.is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.irm_lambda if self.update_count
                              >= self.irm_penalty_anneal_iters else
                              1.0)
        ib_penalty_weight = (self.ib_lambda if self.update_count
                             >= self.ib_penalty_anneal_iters else
                             0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = apply_model(self, all_x).squeeze()
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_x[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.irm_penalty_anneal_iters or self.update_count == self.ib_penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self._init_optimizer()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}
