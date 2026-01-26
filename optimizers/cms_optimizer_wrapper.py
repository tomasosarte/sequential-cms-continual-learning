from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import torch

@dataclass
class CMSGroup:
    params: List[torch.nn.Parameter]
    lr: float                            
    chunk: int = 1                      
    opt_kwargs: Dict[str, Any] | None = None

class CMS:
    def __init__(
        self,
        groups: List[CMSGroup],
        update_fn: Callable[..., None],
    ) -> None:
        self.step_idx: int = 0
        self.groups = groups
        self.update_fn = update_fn
        self.state: Dict[torch.nn.Parameter, Dict[str, Any]] = {}

    @torch.no_grad()
    def step(self) -> None:
        self.step_idx += 1

        for g in self.groups:
            if self.step_idx % g.chunk != 0:
                continue

            for p in g.params:
                st = self.state.setdefault(p, {})
                self.update_fn(
                    param=p,
                    state=st,
                    lr=g.lr,
                    **(g.opt_kwargs or {})
                )

    def zero_grad(self) -> None:
        for g in self.groups:
            if self.step_idx % g.chunk != 0:
                continue
            for p in g.params:
                p.grad = None


def adam_update(
    param: torch.nn.Parameter,
    state: Dict[str, Any],
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
) -> None:

    if param.grad is None:
        return

    grad: torch.Tensor = param.grad

    if weight_decay != 0.0:
        grad = grad.add(param, alpha=weight_decay)

    if not state:
        state["t"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)
        if amsgrad:
            state["vmax"] = torch.zeros_like(param)

    state["t"] += 1
    t: int = state["t"]

    state["m"].mul_(b1).add_(grad, alpha=1 - b1)
    state["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)

    m_hat: torch.Tensor = state["m"] / (1 - b1 ** t)

    if amsgrad:
        state["vmax"].copy_(torch.maximum(state["vmax"], state["v"]))
        v_hat: torch.Tensor = state["vmax"] / (1 - b2 ** t)
    else:
        v_hat = state["v"] / (1 - b2 ** t)

    param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


# import torch
# from torch.optim import Optimizer
# from dataclasses import dataclass
# from typing import List, Type, Dict, Optional

# @dataclass
# class CMSGroup:
#     params: List[torch.nn.Parameter]
#     lr: float
#     chunk: int

# class CMSOptimizerWrapper:
#     def __init__(
#         self,
#         groups: List[CMSGroup],
#         base_optim_cls: Type[Optimizer],
#         base_optim_kwargs: Optional[Dict] = None,
#     ):
#         self.step_idx = 0
#         self.groups = groups
#         base_optim_kwargs = base_optim_kwargs or {}
#         self.opts = [base_optim_cls(g.params, lr=g.lr, **base_optim_kwargs) for g in groups]
        
#     @torch.no_grad()
#     def step(self):
#         self.step_idx += 1
#         for opt, g in zip(self.opts, self.groups):
#             if (self.step_idx % g.chunk) == 0:
#                 opt.step()

#     def zero_grad(self):
#         for opt, g in zip(self.opts, self.groups):
#             if ((self.step_idx)% g.chunk) == 0:
#                 opt.zero_grad()
        