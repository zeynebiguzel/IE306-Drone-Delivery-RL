"""
Small manual Adam optimizer for Role B.

This avoids relying on torch.optim.Adam, which may require extra optional
dependencies in some PyTorch installations.
"""

from __future__ import annotations

from typing import Iterable

import torch


class ManualAdam:
    """
    Minimal Adam optimizer implementation.

    It supports the two methods used in train_a2c.py:
    - zero_grad()
    - step()
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue

            grad = p.grad

            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)