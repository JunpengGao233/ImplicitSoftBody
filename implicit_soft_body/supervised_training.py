import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from robot_model import SpringRobot
from network import MLP
from visualization import render_robot


__CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if __name__ == '__main__':
    trajectories = []
    device = 'cpu'
    robot = SpringRobot('cpu')

    num_epochs = 100
    num_frames = 100
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    torch.random.manual_seed(42)
    network = MLP(4*input_size, output_size)
    # network = model
    # a = torch.randn((output_size, num_frames), requires_grad=True)
    network = network.to(device)
    # optimizer = torch.optim.Adam(network.parameters(),lr=1e-2,weight_decay=1e-3,maximize=True)
    optimizer = torch.optim.Adam(network.parameters(),lr=1e-2,weight_decay=1e-4)#maximize=True)
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
                traj_file_path = os.path.join(__CURRENT_PATH,  f"trajectory/{i}.json")
                with open(traj_file_path) as f:
                    traj_dict = json.load(f)
                    x_target = torch.tensor(traj_dict['pos1'])
                    x0 = torch.tensor(traj_dict['pos0'])
                    v0 = torch.tensor(traj_dict['vel0'])
                    a1 = torch.tensor(traj_dict['a1'])
                a = network(torch.cat([x0.flatten()-robot.x0.flatten(),v0.flatten()], dim=0))
                a = 0.4 * torch.nn.functional.tanh(a) + 0.6
                # x, v = robot.forward(x0, v0, a)
                loss += (a1- a).pow(2).mean()
                # loss += (x_target - x).pow(2).mean()
                actuation_seq.append(a.detach().cpu().numpy())
            loss.backward(retain_graph=True)
            # print(a0.grad)
            return loss
    
        loss = optimizer.step(closure)
        # loss = closure()
        # make_dot(loss).render("attached", format="png")
        with np.printoptions(precision=3):
            print(f'training epoch {epoch}: loss {loss.item()}, relative loss change {torch.abs((loss-loss_last)/loss).item()}')
        # if torch.abs((loss-loss_last)/loss) < 1e-4:
        #     break
        loss_history.append(loss.item())
        loss_last = loss
        if loss >= np.max(loss_history):
            print("saving best, loss:", loss)
            np.save('actuation_seq_best.npy', actuation_seq)
            torch.save(
                network.state_dict(),
                f"network_best.pth",
            )
        # print(actuation_seq)
    actuation_seq = np.array(actuation_seq)
    output = render_robot(actuation_seq)
    with open(f'final{epoch}.html', 'w') as f:
        f.write(output)
    x = robot.x
    v = robot.v
    actuation_seq = np.concatenate([actuation_seq, actuation_seq], axis=0)
    for i in range(actuation_seq.shape[0]):
        a = torch.tensor(actuation_seq[i])
        x, v = robot.forward(x, v, a)
    robot.x1 = x
    robot.v1 = v
    
    # x2 = robot.forward(robot.x)
    num_epochs = 200
    num_frames = 50
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    torch.random.manual_seed(42)
    network2 = MLP(4*input_size, output_size, 32)
    # add pretrain to model
    # network = model
    network = network.to(device)
    # pretrained model
    # network.load_state_dict(torch.load("policy.pt"))
    optimizer = torch.optim.Adam(network2.parameters(),lr=1e-2)#, maximize=True)
    loss_last = 1e10
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        actuation_seq2 = []
        def closure():
            optimizer.zero_grad()
            x = robot.x1
            v = robot.v1
            a = torch.ones_like(robot.l0)
            last_p = robot.x_pos(x)
            loss = 0
            for i in range(num_frames):
                x = x.to(device)
                v = v.to(device)
                a = a.to(device)
                a = network2(torch.cat([x.flatten()-robot.x.flatten(),v.flatten()], dim=0))
                a = 0.4 * torch.nn.functional.tanh(a) + 0.6
                x, v = robot.forward(x, v, a)
                actuation_seq2.append(a.detach().cpu().numpy())
                loss += (x - robot.x).pow(2).mean()
            loss.backward()
            return loss
    
        loss = optimizer.step(closure)
        # loss = closure()
        # make_dot(loss).render("attached", format="png")
        with np.printoptions(precision=3):
            print(f'epoch {epoch}: loss {loss.item()}, relative loss change {torch.abs((loss-loss_last)/loss).item()}')
        # if torch.abs((loss-loss_last)/loss) < 1e-4:
        #     break
        loss_history.append(loss.item())
        loss_last = loss
        if loss <= np.min(loss_history):
            print("saving best, loss:", loss)
            model_dict = network.state_dict()
            torch.save(model_dict, "policy_train.pt")
            np.save('actuation_seq_best.npy', actuation_seq)
        # print(actuation_seq)
        actuation_seq = np.array(actuation_seq)
        actuation_seq2 = np.array(actuation_seq2)
        print(actuation_seq.shape)
        print(actuation_seq2.shape)
        actuation_seq_n = np.concatenate([actuation_seq, actuation_seq2], axis=0)
        output = render_robot(actuation_seq_n)
        with open(f'stand{epoch}.html', 'w') as f:
            f.write(output)


    actuation_seq = np.array(actuation_seq)
    np.save('actuation_seq.npy', actuation_seq) 
    np.save('loss_history.npy', loss_history)


