import torch
import torch.nn.functional as F
import numpy as np

import system
from network import MLP
from load_weights import model, preprocess, postprocess
from visualization import render_robot

from torchviz import make_dot
class SpringRobot(system.MassSpringSystem):
    def __init__(self,  device='cpu') -> None:
        self.x = torch.tensor( [
          [0, 0],
          [1, 1]
        ], dtype=torch.float32)
        self.springs = torch.tensor([[0,1]])
        self.triangles = torch.tensor([[0,1,1]])
        self.l0 = torch.ones([1]) * np.sqrt(2)
        params = {
            "mass": 6.0714287757873535,
            "k_spring": 90,
            "l0": self.l0,
            "mu": 500,
            "nu": 50,
            "k_collision": 14000,
            "k_friction": 300,
            "epsilon": 0.01,
            "dt": 0.033,
            "max_iter": 100,
        }
        super().__init__(self.x, self.springs, self.triangles, params, device)
    

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = 'cpu'
    robot = SpringRobot(device)
    num_epochs = 100
    num_frames = 1#00
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    x_target = torch.tensor([0.9, 0.9])
    torch.random.manual_seed(42)
    # da = (torch.randn_like(robot.l0) * 1e-2).requires_grad_()
    # da = torch.tensor([0.2]).requires_grad_()
    a = (torch.ones_like(robot.l0)*0.8).requires_grad_(True)
    optimizer = torch.optim.Adam([a],lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_last = 1e10
    for epoch in range(num_epochs):
        print("-------------------")
        actuation_seq = []
        optimizer.zero_grad()
        x = robot.x
        v = robot.v
        v = torch.zeros_like(robot.v)
        loss = 0
        for i in range(num_frames):
            # a = a - da
            print("a", a)
            x, v = robot.forward(x, v, a)
            actuation_seq.append(a.detach().cpu().numpy())
        loss = (x_target - x[1]).pow(2).sum() #+ torch.norm(x[0])**2
        print("loss\n", loss, "\nx\n", x[1])
        loss.backward()
        optimizer.step()
        scheduler.step()
        with np.printoptions(precision=3):
            print(f'epoch {epoch}: loss {loss.item()}, relative loss change {torch.abs((loss-loss_last)/loss).item()}')
        loss_history.append(loss.item())
        loss_last = loss
        if loss <= np.min(loss_history):
            print("saving best, loss:", loss)
            np.save('actuation_seq_best.npy', actuation_seq)
        actuation_seq = np.array(actuation_seq)
        output = render_robot(actions=actuation_seq,pos=robot.x.detach().cpu().numpy(), triangles=robot.triangles.detach().numpy(), muscles=robot.springs.detach().numpy() )
        with open(f'debug{epoch}.html', 'w') as f:
            f.write(output)
        # print(actuation_seq)
    np.save('actuation_seq.npy', actuation_seq) 

