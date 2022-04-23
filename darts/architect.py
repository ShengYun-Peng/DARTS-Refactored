import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from typing import List

class Architect(object):
    """
    Update arch parameters alpha
    """
    def __init__(
        self,
        model: nn.Module,
        args: argparse.Namespace,
    ) -> None:
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _concat(self, xs: torch.Tensor) -> torch.Tensor:
        """
        flattern all the tensors and concat in a 1-d tensor
        """
        out = torch.cat([x.view(-1) for x in xs])
        return out

    def step(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float,
        network_optimizer: nn.Module,
        unrolled: bool,
    ):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid,
                eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        self.optimizer.step()

    def _backward_step_unrolled(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float,
        network_optimizer: nn.Module,
    ):
        # prepared unrolled model with updated weights
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        # compute L_{val}(w', alpha)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()

        # d(L_{val}(w', alpha)) / d(alpha), first half of eq. 7
        dalpha = list([v.grad for v in unrolled_model.arch_parameters()])
        # d(L_{val}(w', alpha)) / d(w')
        vector = list([v.grad.data for v in unrolled_model.parameters()])

        # implicit grads are the second half of eq. 7
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # compute the full arch grad in eq. 7
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                g.data.requires_grad_()
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(
        self,
        vector: torch.Tensor,
        input: torch.Tensor,
        target: torch.Tensor,
        r: float=1e-2,
    ) -> List:
        R = r / self._concat(vector).norm()  # epsilon, a small scalar defined in the paper

        # w + epsilon d(L_{val}(w', alpha)) / d(w'), self.model.parameters are not updated
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        # d(L_{train}(w+, alpha)) / d(alpha)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        # w - epsilon d(L_{val}(w', alpha)) / d(w')
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)

        # d(L_{train}(w-, alpha)) / d(alpha)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # restore parameters
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        # second half of the eq. 7, which is also eq. 8 in the paper
        out = list([(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)])
        return out


    def _compute_unrolled_model(
        self,
        input: torch.Tensor,  # training input
        target: torch.Tensor,
        eta: float, # learning rate
        network_optimizer: nn.Module
    ) -> nn.Module:
        # compute training loss
        loss = self.model._loss(input, target)

        # theta corresponds to the weights
        theta = self._concat(self.model.parameters()).data
        try:
            moment = self._concat(
                network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        dtheta = self._concat(torch.autograd.grad(loss, self.model.parameters())).data
        dtheta += self.network_weight_decay * theta

        # w - lr * d(L_{train}(w, alpha)) / d(w) in formula 6
        # prepare the updated weights model for val loss computation
        theta = theta.sub(moment + dtheta, alpha=eta)

        unrolled_model = self._construct_model_from_theta(theta)
        return unrolled_model

    def _construct_model_from_theta(self, theta: torch.Tensor) -> nn.Module:
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params = dict()
        offset = 0  # since theta is flatterned, we need an offset to fetch the weights
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.shape)
            params[k] = theta[offset: offset + v_length].reshape(v.shape)
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        model_new = model_new.cuda()

        return model_new

    def _backward_step(self, input_valid: torch.Tensor, target_valid: torch.Tensor):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

