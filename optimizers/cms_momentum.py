import torch
from torch.optim import Optimizer


class CMSMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta_fast=0.9,
        beta_slow=0.95,
        slow_period=4,
        lam=0.7,
        use_multi_timescale = True,
    ):
        defaults = dict(
            lr=lr,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            slow_period=slow_period,
            lam=lam,
            use_multi_timescale=use_multi_timescale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta_f = group["beta_fast"]
            beta_s = group["beta_slow"]
            slow_p = group["slow_period"]
            lam = group["lam"]
            use_multi_timescale = group["use_multi_timescale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # init state
                if len(state) == 0:
                    state["step"] = 0
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)
                    if use_multi_timescale:
                        slow_mem = torch.zeros_like(p.grad)
                
                state["step"] += 1
                t = state["step"]
                m_fast = state["m_fast"]
                m_slow = state["m_slow"]
                
                # 1. Update fast momentum
                m_fast.mul_(beta_f).add_(grad)

                if use_multi_timescale:
                    slow_mem = state["slow_mem"]

                    # 2. Update slow momentum
                    if t % slow_p == 0:
                        m_slow.mul_(beta_s).add_(slow_mem/slow_p) # Should I divide the gradient by the number of periods?
                        slow_mem.zero_()
                    else:
                        slow_mem.add_(grad)

                    # 3. Combine momentums
                    combination = lam * m_fast + (1 - lam) * m_slow
                    p.add_(combination, alpha=-lr)
                
                else:
                    p.add_(m_fast, alpha=-lr)