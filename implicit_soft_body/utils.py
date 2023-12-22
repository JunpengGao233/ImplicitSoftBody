import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from implicit_soft_body import IMPLICIT_SOFT_BODY_POLICY_ROOT

policy_file_path = os.path.join(IMPLICIT_SOFT_BODY_POLICY_ROOT,  "pretrained_policy.json")

with open(policy_file_path) as f:
    policy_dict = json.load(f)

policy_dict:dict


__min_a = policy_dict['min_a']  # 0.25
max_abs_da = policy_dict['max_abs_da']  # 0.3 
center_vertex_id = policy_dict['center_vertex_id']  # 8
forward_vertex_id = policy_dict['forward_vertex_id']  # 6


def postprocess(action:torch.Tensor, action_change:torch.Tensor,max_abs_da:float=max_abs_da, min_a:float=__min_a, max_a:float=1.0,noise_std:Optional[float]= None):
    if noise_std is not None:
        action_change = action_change +  noise_std * torch.randn_like(action_change)
    action_change = torch.clamp(action_change, min=-max_abs_da, max=max_abs_da)
    
    action = action + action_change
    action = torch.clamp(action, min=min_a, max=max_a)
    return action
    

def frame_project_pos(pos:torch.Tensor, center_id:int, forward_id:int, relative:bool=False)-> torch.Tensor:
    center_ = pos[center_id, :]
    forward_ = pos[forward_id, :]
    
    vector_0 = forward_ - center_
    vector_0 = F.normalize(vector_0, dim=0)

    rotation = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=vector_0.device)
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
    vector_0 = F.normalize(vector_0, dim=0).to(vector_0.device)

    rotation = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32,device=vector_0.device)
    # print(rotation.device)
    vector_1 = rotation @ vector_0

    absolute_x = vel @ vector_0
    absolute_y = vel @ vector_1
    return absolute_x, absolute_y

def preprocess(pos:torch.Tensor, vel:torch.Tensor, center_id:int, forward_id:int)->torch.Tensor:
    pos_x, pos_y = frame_project_pos(pos, center_id=center_id, forward_id=forward_id, relative=True)
    pos_output = torch.hstack((pos_x[:, None], pos_y[:, None])).flatten()
    vel_x, vel_y = frame_project_vel(pos, vel, center_id=center_id, forward_id=forward_id, relative=False)
    vel_output = torch.hstack((vel_x[:, None], vel_y[:, None])).flatten()
    
    return torch.concat((pos_output, vel_output), dim=0)
    
