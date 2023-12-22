"""
This script pretrains a model to imitate the acutation controller sequence of the robot.
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
import numpy as np
import os
from implicit_soft_body.network import MLP
from implicit_soft_body import IMPLICIT_SOFT_BODY_ROOT
from implicit_soft_body import IMPLICIT_SOFT_BODY_POLICY_ROOT

output_range = torch.tensor([0.2, 1.2])


# load data

dataset_dir = os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "assets", "dataset")

input_history = np.load(os.path.join(dataset_dir, "input_history_pretrain.npy"))
actuation_seq = np.load(os.path.join(dataset_dir, "actuation_seq_pretrain.npy"))

input_history = torch.tensor(input_history)
input_size = input_history.shape[1]
actuation_seq = torch.tensor(actuation_seq)
actuation_size = actuation_seq.shape[1]

# scale the output to be in the range of the training data
actuation_seq = (actuation_seq - torch.mean(output_range)) / (torch.max(output_range) - torch.min(output_range)) * 2



num_epochs = 100

# load data for training

train_data = TensorDataset(input_history, actuation_seq)
train_data = DataLoader(train_data, batch_size=32, shuffle=True)
model = MLP(input_size, actuation_size, hidden_size=32)

# train the model with  MSE loss with any evaluation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
for epoch in range(0, num_epochs):
    for input, a in train_data:
        optimizer.zero_grad()
        a_pred = model(input)
        loss = criterion(a_pred, a)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}, loss {loss.item()}")

torch.save(model.state_dict(), os.path.join(IMPLICIT_SOFT_BODY_POLICY_ROOT, "pretrained_policy.pt"))