from typing import Iterable
import torch
from torch.optim import Optimizer


# ------------------------------------------------------------
# Newton–Schulz orthogonalization (as given)
# ------------------------------------------------------------
@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


# ------------------------------------------------------------
# Muon update primitive
# ------------------------------------------------------------
def muon_update(
        grad: torch.Tensor,
        momentum: torch.Tensor,
        beta: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True
    ) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)

    # TODO: Only apply NS to matrix-like tensors (i.e., ignore biases, BN params, etc.)
    if update.ndim < 2:
        return update

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


# ------------------------------------------------------------
# Minimal Muon Optimizer
# ------------------------------------------------------------
class Muon(Optimizer):
    """
    Minimal Muon optimizer (single device, no extras).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 1. Get state
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                momentum = state["momentum"]

                # 2. Compute Muon update
                update = muon_update(
                    grad=g,
                    momentum=momentum,
                    beta=beta,
                    ns_steps=ns,
                )

                # 3. Apply weight decay
                p.mul_(1 - lr * wd)

                # 4. Parameter update
                p.add_(update.reshape(p.shape), alpha=-lr)

class MultiScaleMuonMyVersion(torch.optim.Optimizer):
    """
    Multi-scale Muon (M3-style)

    Fast Muon  : per-step momentum + NS
    Slow Muon  : block-accumulated momentum + NS
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum_fast: float = 0.95,
        momentum_slow: float = 0.9,
        alpha: float = 1.0,
        frequency: int = 2,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum_fast=momentum_fast,
            momentum_slow=momentum_slow,
            alpha=alpha,
            frequency=frequency,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1

        for group in self.param_groups:
            lr = group["lr"]
            bf = group["momentum_fast"]
            bs = group["momentum_slow"]
            alpha = group["alpha"]
            frequency = group["frequency"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 1. Get state
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)
                    state["g_acc"]  = torch.zeros_like(p)
                    state["slow_update"] = torch.zeros_like(p)
                m_fast = state["m_fast"]
                m_slow = state["m_slow"]
                g_acc  = state["g_acc"]
                slow_u = state["slow_update"]

                # 2. Fast momentum muon update
                fast_update = muon_update(
                    grad=g,
                    momentum=m_fast,
                    beta=bf,
                    ns_steps=ns
                )

                # 3. Slow momentum update
                g_acc.add_(g)
                if self.t % frequency == 0:
                    slow_update = muon_update(
                        grad=g_acc, 
                        momentum=m_slow, 
                        beta=bs, 
                        ns_steps=ns
                    )
                    slow_u.copy_(slow_update)
                    g_acc.zero_()

                # 4. Weight decay
                p.mul_(1 - lr * wd)

                # 5. Parameter update
                p.add_(
                    (fast_update + alpha * slow_u).reshape(p.shape),
                    alpha=-lr,
                )

class MultiScaleMuon(torch.optim.Optimizer):
    """
    Nested Learning M3 optimizer
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        lr_m1: float = 0.9,
        lr_m2: float = 0.9,
        lr_v1: float = 0.9,
        alpha: float = 1.0,
        frequency: int = 2,
        ns_steps: int = 5,
        epsilon: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            lr_m1=lr_m1,
            lr_m2=lr_m2,
            lr_v1=lr_v1,
            alpha=alpha,
            frequency=frequency,
            ns_steps=ns_steps,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1

        for group in self.param_groups:
            lr = group["lr"]
            lr_m1 = group["lr_m1"]
            lr_m2 = group["lr_m2"]
            lr_v1 = group["lr_v1"]
            alpha = group["alpha"]
            frequency = group["frequency"]
            ns_steps = group["ns_steps"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 1. Get state
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v1"] = torch.zeros_like(p)
                    state["g_acc"]  = torch.zeros_like(p)
                    state["o2"] = torch.zeros_like(p)
                
                m1, m2, v1 = state["m1"], state["m2"], state["v1"]
                g_acc  = state["g_acc"]
                

                # 2. Fast momentum muon update
                m1.add_(lr_m1 * g)
                v1.add_(lr_v1 * g * g)
                o1 = zeropower_via_newtonschulz5(m1, ns_steps) if m1.ndim >= 2 else m1

                # 3. Slow momentum update
                if self.t % frequency == 0:
                    m2.add_(lr_m2 * g_acc)
                    state["o2"].copy_(
                        zeropower_via_newtonschulz5(m2, ns_steps) if m2.ndim >= 2 else m2
                    )
                    g_acc.zero_()
                g_acc.add_(g)
                o2 = state["o2"]

                # 4. Parameter update
                denominator = v1.sqrt().add_(epsilon)
                p.add_(-lr * ((o1 + alpha * o2) / denominator))