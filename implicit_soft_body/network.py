import torch
from torch import nn

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, input_size,output_size,hidden_size=32):
        super().__init__()
        model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Tanh()
        )

        self.network = model

    def forward(self, x):
        """Forward pass"""
        x = self.network(x)
        x = torch.clamp(x, -0.3, 0.3)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)