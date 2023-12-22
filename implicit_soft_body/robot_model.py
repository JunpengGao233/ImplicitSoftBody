import torch
import torch.nn.functional as F
import numpy as np
from implicit_soft_body import IMPLICIT_SOFT_BODY_MESH_ROOT
import os
from typing import Union
from .system import MassSpringSystem
from .network import MLP
from .utils import  preprocess, postprocess
import json 

# from torchviz import make_dot
class SpringRobot(MassSpringSystem):
    def __init__(self,  device:Union[str, torch.device]='cpu') -> None:
        self.x = torch.tensor([[1.6825000047683716, 0.5799999833106995], [1.6675000190734863, 0.9549999833106995], [1.315000057220459, 0.9399999976158142], [1.472499966621399, 0.7749999761581421], [1.2100000381469727, 0.4449999928474426], [1.375, 0.32499998807907104], [1.4800000190734863, 1.1649999618530273], [1.225000023841858, 1.4500000476837158], [1.75, 1.4199999570846558], [1.600000023841858, 1.8700000047683716], [2.5, 2.049999952316284], [2.424999952316284, 1.5850000381469727], [2.005000114440918, 0.8949999809265137], [2.755000114440918, 1.5700000524520874], [3.0850000381469727, 1.4950000047683716], [2.7325000762939453, 0.9850000143051147], [2.4024999141693115, 0.925000011920929], [3.077500104904175, 0.9700000286102295], [3.047499895095825, 0.6549999713897705], [2.882499933242798, 0.5950000286102295], [1.5700000524520874, 0.2199999988079071], [1.1349999904632568, 0.009999999776482582], [3.0325000286102295, 0.2800000011920929], [3.302500009536743, 0.29499998688697815], [2.8524999618530273, 0.02500000037252903], [3.497499942779541, 0.12999999523162842], [1.7350000143051147, 0.009999999776482582], [2.0950000286102295, 1.6150000095367432]])
        self.triangles = torch.tensor([[1, 2, 3], [1, 3, 0], [3, 5, 4], [3, 0, 5], [3, 2, 4], [6, 7, 2], [1, 6, 2], [8, 9, 6], [9, 6, 7], [8, 1, 6], [12, 8, 1], [10, 11, 13], [14, 10, 13], [15, 14, 13], [16, 15, 11], [15, 11, 13], [17, 14, 15], [18, 19, 15], [18, 15, 17], [20, 21, 5], [21, 5, 4], [22, 23, 18], [19, 22, 18], [17, 23, 18], [24, 22, 23], [5, 0, 20], [25, 23, 24], [22, 24, 19], [26, 20, 21], [27, 8, 12], [27, 11, 16], [27, 12, 16], [9, 27, 8], [27, 11, 10], [9, 10, 27]])
        self.springs = torch.tensor([[2, 4], [8, 1], [1, 12], [12, 8], [15, 11], [15, 16], [16, 11], [18, 17], [18, 23], [17, 23], [0, 20], [5, 0], [5, 20], [22, 19], [22, 24], [19, 24], [12, 16], [3, 2], [4, 3]])
        self.l0 = torch.tensor([0.5060138702392578, 0.47226181626319885, 0.34279194474220276, 0.5836523175239563, 0.6742079257965088, 0.33541035652160645, 0.6603834629058838, 0.3164254426956177, 0.4411633014678955, 0.7115125060081482, 0.3771687150001526, 0.39947620034217834, 0.2214723825454712, 0.34889116883277893, 0.31212979555130005, 0.5707889795303345, 0.3986302614212036, 0.22810354828834534, 0.42167073488235474])
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


class SimpleRobot(MassSpringSystem):
    def __init__(self,  device:Union[str, torch.device]='cpu') -> None:
        assert IMPLICIT_SOFT_BODY_MESH_ROOT is not None
        robot_mesh_path = os.path.join(IMPLICIT_SOFT_BODY_MESH_ROOT, "simple_mesh.json")
        with open(robot_mesh_path) as f:
            json_dict = json.load(f)
        self.x = torch.tensor(json_dict['pos'])
        self.triangles = torch.tensor(json_dict['triangles'])
        self.springs = torch.tensor(json_dict['springs'])
        self.l0 = torch.linalg.norm(self.x[self.springs[:,0], :] - self.x[self.springs[:,1], :], dim=1)

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
    def x_pos(self, x: torch.Tensor):
        return x[1, 0]
    

class LeggedRobot(MassSpringSystem):
    def __init__(self,  device:Union[str, torch.device]='cpu') -> None:
        assert IMPLICIT_SOFT_BODY_MESH_ROOT is not None
        robot_mesh_path = os.path.join(IMPLICIT_SOFT_BODY_MESH_ROOT, "normalized_mesh_0.json")
        with open(robot_mesh_path) as f:
            json_dict = json.load(f)
        self.x = torch.tensor(json_dict['pos'])
        self.triangles = torch.tensor(json_dict['triangles'])
        self.springs = torch.tensor(json_dict['springs'])
        self.l0 = torch.linalg.norm(self.x[self.springs[:,0], :] - self.x[self.springs[:,1], :], dim=1)

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
    def x_pos(self, x: torch.Tensor):
        return x[1, 0]
    
class GenericRobot(MassSpringSystem):
    def __init__(self,mesh_path:str, device:Union[str, torch.device]='cpu') -> None:
        with open(mesh_path) as f:
            json_dict = json.load(f)
        self.x = torch.tensor(json_dict['pos'])
        self.triangles = torch.tensor(json_dict['triangles'])
        self.springs = torch.tensor(json_dict['springs'])
        self.l0 = torch.linalg.norm(self.x[self.springs[:,0], :] - self.x[self.springs[:,1], :], dim=1)

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
    def x_pos(self, x: torch.Tensor):
        return x[1, 0]
    

if __name__ == '__main__':
    # Test the training in the simulator 
    # The same script can be found in 'scripts/train.py'
    min_a = 0.25
    max_abs_da = 0.3
    center_vertex_id = 8
    forward_vertex_id = 6

    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = 'cpu'
    robot = SimpleRobot(device=device)


    # x2 = robot.forward(robot.x)
    num_epochs = 30
    num_frames = 10
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    torch.random.manual_seed(42)
    network = MLP(4*input_size, output_size, 24)
    # add pretrain to model
    # network = model
    network = network.to(device)
    # pretrained model
    # network.load_state_dict(torch.load("policy.pt"))
    optimizer = torch.optim.Adam(network.parameters(),lr=5e-4)
    loss_last = 1e10
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        actuation_seq = []
        def closure():
            optimizer.zero_grad()
            x = robot.x
            v = robot.v
            a = 1 / robot.l0 + 0.3 * torch.randn_like(robot.l0)
            last_p = robot.x_pos(x)
            loss = 0
            for i in range(num_frames):
                x = x.to(device)
                v = v.to(device)
                a = a.to(device)
                da = network(preprocess(x,v, center_id=center_vertex_id, forward_id=forward_vertex_id))
                a = postprocess(a, da, noise_std=0.15, min_a=min_a, max_a=1.0)
                x, v = robot.forward(x, v, a)
                actuation_seq.append(a.detach().cpu().numpy())
                curr_p = robot.x_pos(x)
                loss += curr_p  - last_p
                last_p = curr_p
                
            # loss /= num_frames
            loss *= -1
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
            torch.save(model_dict, os.path.join(output_path, "policy_train.pt"))
            np.save(os.path.join(output_path, "loss_history_best.npy"), loss_history)
        # print(actuation_seq)
    model_dict = network.state_dict()
    loss_history = np.array(loss_history)
    np.save(os.path.join(output_path, "loss_history.npy"), loss_history)
    torch.save(model_dict, os.path.join(output_path, "policy_train_last.pt"))
    actuation_seq = np.array(actuation_seq)
    np.save(os.path.join(output_path, "actuation_seq.npy"), actuation_seq)

