import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
policy_file_path = os.path.join(__CURRENT_PATH,  "policy.json")

with open(policy_file_path) as f:
    policy_dict = json.load(f)

policy_dict:dict

fc1 = policy_dict['fc1']
fc1_weights = fc1["weight"]
fc1_bias = fc1["bias"]
fc2 = policy_dict['fc2']
fc2_weights = fc2["weight"]
fc2_bias = fc2["bias"]
min_a = policy_dict['min_a']
max_abs_da = policy_dict['max_abs_da']
center_vertex_id = policy_dict['center_vertex_id']
forward_vertex_id = policy_dict['forward_vertex_id']

class MLP(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x

fc1_weights_np = np.array(fc1_weights)
# print(fc1_weights_np.shape)
fc2_weights_np = np.array(fc2_weights)
# print(fc2_weights_np.shape)
        
model = MLP(fc1_weights_np.shape[1], fc1_weights_np.shape[0], fc2_weights_np.shape[0])

state_dict = {
    "fc1.weight": torch.tensor(fc1_weights, requires_grad=True),
    "fc1.bias":torch.tensor(fc1_bias, requires_grad=True),
    "fc2.weight": torch.tensor(fc2_weights, requires_grad=True),
    "fc2.bias":torch.tensor(fc2_bias, requires_grad=True),
}
model.load_state_dict(state_dict)

test_tensor = torch.ones((fc1_weights_np.shape[1],))
predict = model(test_tensor)


def postprocess(action:torch.Tensor, action_change:torch.Tensor, max_abs_da:float=max_abs_da, min_a:float=min_a):
    action_change = torch.clamp(action_change, min=-max_abs_da, max=max_abs_da)
    
    action = action + action_change
    action = torch.clamp(action, min=min_a, max=1.0)
    return action
    

def frame_project_pos(pos:torch.Tensor, center_id:int, forward_id:int, relative:bool=False)-> torch.Tensor:
    center_ = pos[center_id, :]
    forward_ = pos[forward_id, :]
    
    vector_0 = forward_ - center_
    vector_0 = F.normalize(vector_0, dim=0)

    rotation = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
    vector_1 = rotation @ vector_0

    if relative:
        relative_ = pos - center_
        relative_x = relative_ @ vector_0
        relative_y = relative_ @ vector_1
        return relative_x, relative_y
    absolute_x = pos @ vector_0
    absolute_y = pos @ vector_1
    return absolute_x, absolute_y

def frame_project_vel(pos:torch.Tensor, vel:torch.Tensor, center_id:int, forward_id:int, relative:bool=False)-> torch.Tensor:
    center_ = pos[center_id, :]
    forward_ = pos[forward_id, :]
    
    vector_0 = forward_ - center_
    vector_0 = F.normalize(vector_0, dim=0)

    rotation = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
    vector_1 = rotation @ vector_0

    absolute_x = vel @ vector_0
    absolute_y = vel @ vector_1
    return absolute_x, absolute_y

def preprocess(pos:torch.Tensor, vel:torch.Tensor, center_id:int=center_vertex_id, forward_id:int=forward_vertex_id)->torch.Tensor:
    pos_x, pos_y = frame_project_pos(pos, center_id=center_id, forward_id=forward_id, relative=True)
    pos_output = torch.hstack((pos_x[:, None], pos_y[:, None])).flatten()
    vel_x, vel_y = frame_project_vel(pos, vel, center_id=center_id, forward_id=forward_id, relative=False)
    vel_output = torch.hstack((vel_x[:, None], vel_y[:, None])).flatten()
    
    return torch.concat((pos_output, vel_output), dim=0)
    
