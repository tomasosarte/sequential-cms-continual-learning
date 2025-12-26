import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
        input_dim=784,
        hidden_dims=[256, 128, 64],
        output_dim=10,
        ):
        super().__init__()
        assert len(hidden_dims) == 3
        self.slow = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        self.mid = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.fast = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim),
        )

    def forward(self, x):
        h = self.slow(x)
        h = self.mid(h)
        y = self.fast(h)
        return y