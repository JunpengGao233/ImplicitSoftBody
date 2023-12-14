import torch


class MLP(torch.nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, input_size, output_size, act_fn=torch.nn.ELU()):
        super().__init__()
        hidden_sizes = [
            input_size,
            output_size,
            # output_size,
            # output_size,
        ]
        layers = []
        for i in range(len(hidden_sizes) - 1):
            cur_layer = torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            layers.append(cur_layer)
            layers.append(act_fn)
        layers = layers[:-1]

        self.hidden_sizes = hidden_sizes
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        x = self.network(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)