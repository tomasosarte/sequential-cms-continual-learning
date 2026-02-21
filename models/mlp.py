import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
        input_dim=784,
        hidden_dims=[256, 128, 64],
        output_dim=10,
        activation=nn.ReLU
        ):
        
        super().__init__()
        self.levels = nn.ModuleList()
        self.levels.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                activation(),
            )
        )
        for i in range(len(hidden_dims) - 1):
            if i != len(hidden_dims) - 2:
                self.levels.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        activation(),
                    )
                )
            else:
                self.levels.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        activation(),
                        nn.Linear(hidden_dims[i+1], output_dim),
                    )
                )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.levels:
            x = layer(x)
        return x