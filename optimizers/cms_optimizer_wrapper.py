import torch
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import List, Type, Dict, Optional

@dataclass
class CMSGroup:
    params: List[torch.nn.Parameter]
    lr: float
    chunk: int

class CMSOptimizerWrapper:
    def __init__(
        self,
        groups: List[CMSGroup],
        base_optim_cls: Type[Optimizer],
        base_optim_kwargs: Optional[Dict] = None,
    ):
        self.step_idx = 0
        self.groups = groups
        base_optim_kwargs = base_optim_kwargs or {}
        self.opts = [base_optim_cls(g.params, lr=g.lr, **base_optim_kwargs) for g in groups]
        
    @torch.no_grad()
    def step(self):
        self.step_idx += 1
        for opt, g in zip(self.opts, self.groups):
            if (self.step_idx % g.chunk) == 0:
                opt.step()

    def zero_grad(self):
        for opt, g in zip(self.opts, self.groups):
            if ((self.step_idx)% g.chunk) == 0:
                opt.zero_grad()
        