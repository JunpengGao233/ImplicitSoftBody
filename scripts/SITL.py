import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from implicit_soft_body.robot_model import SpringRobot
from implicit_soft_body.network import MLP
from implicit_soft_body.visualization import render_robot


__CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if __name__ == '__main__':
    trajectories = []
    device = 'cpu'
    robot = SpringRobot('cpu')

    num_epochs = 40
    num_frames = 100
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    torch.random.manual_seed(42)
    network = MLP(4*input_size, output_size)
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_last = 0
    prepare_steps = 10
    for epoch in range(num_epochs):
        actuation_seq = []
        print(f'epoch {epoch}')
        def closure():
            optimizer.zero_grad()
            x = robot.x
            v = robot.v
            loss = 0
            for i in range(num_frames):
                x = x.to(device)
                v = v.to(device)
                traj_file_path = os.path.join(__CURRENT_PATH,  f"../assets/trajectory/{i}.json")
                with open(traj_file_path) as f:
                    traj_dict = json.load(f)
                    x_target = torch.tensor(traj_dict['pos1'])
                    x0 = torch.tensor(traj_dict['pos0'])
                    v0 = torch.tensor(traj_dict['vel0'])
                a = network(torch.cat([x0.flatten()-robot.x0.flatten(),v0.flatten()], dim=0))
                a = 0.4 * torch.nn.functional.tanh(a) + 0.6
                x, v= robot.forward(x, v, a)
                loss += (x_target - x).pow(2).mean()
                actuation_seq.append(a.detach().cpu().numpy())
            loss /= num_frames
            loss.backward()
            # print(a0.grad)
            return loss
    
        loss = optimizer.step(closure)
        with np.printoptions(precision=3):
            print(f'training epoch {epoch}: loss {loss.item()}, relative loss change {torch.abs((loss-loss_last)/loss).item()}')
        loss_history.append(loss.item())
        loss_last = loss
        if loss <= np.min(loss_history):
            print("saving best, loss:", loss)
            np.save('actuation_seq_best.npy', actuation_seq)
            torch.save(
                network.state_dict(),
                f"network_best.pth",
            )
        actuation_seq = np.array(actuation_seq)
        output = render_robot(actuation_seq)
        with open(f'SITL{epoch}.html', 'w') as f:
            f.write(output)