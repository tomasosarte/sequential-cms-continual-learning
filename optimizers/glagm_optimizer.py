import torch
from torch.optim import Optimizer

class GLAGM(Optimizer):
    """
    Global Loss-Aligned Gradient Memory (minimal version)
    """

    def __init__(self, params, lr=1e-3, mu=0.9, lam=0.5):
        defaults = dict(lr=lr, mu=mu, lam=lam)
        super().__init__(params, defaults)
        self.t = 0  # global step counter

    @torch.no_grad()
    def step(self):
        self.t += 1

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # initialize state
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["a"] = torch.zeros_like(p)

                m = state["m"]
                a = state["a"]

                # momentum (plasticity)
                # m.mul_(mu).add_(grad, alpha=1 - mu)
                # momentum velocity version
                m.mul_(mu).add_(grad)

                # running average (stability / global objective)
                a.mul_((self.t - 1) / self.t).add_(grad, alpha=1 / self.t)

                # combined direction
                d = lam * m + (1 - lam) * a

                # parameter update
                p.add_(d, alpha=-lr)
