import torch
from torch import nn
from implicit_soft_body import IMPLICIT_SOFT_BODY_POLICY_ROOT
import os
import json

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, input_size,output_size,hidden_size=112):
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
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    

__policy_file_path = os.path.join(IMPLICIT_SOFT_BODY_POLICY_ROOT,  "pretrained_policy.json")

with open(__policy_file_path) as f:
    __policy_dict = json.load(f)

__policy_dict:dict

__fc1 = __policy_dict['fc1']
__fc1_weights = __fc1["weight"]
__fc1_bias = __fc1["bias"]
__fc2 = __policy_dict['fc2']
__fc2_weights = __fc2["weight"]
__fc2_bias = __fc2["bias"]

__state_dict = {
    "network.0.weight": torch.tensor(__fc1_weights, requires_grad=True),
    "network.0.bias":torch.tensor(__fc1_bias, requires_grad=True),
    "network.2.weight": torch.tensor(__fc2_weights, requires_grad=True),
    "network.2.bias":torch.tensor(__fc2_bias, requires_grad=True),
}

# Load the model trained by the author of the paper
model = MLP(__state_dict["network.0.weight"].shape[1], __state_dict["network.2.weight"].shape[0], __state_dict["network.0.weight"].shape[0])
model.load_state_dict(__state_dict)