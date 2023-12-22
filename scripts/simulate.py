from implicit_soft_body.robot_model import SimpleRobot
from implicit_soft_body.utils import postprocess, preprocess
from implicit_soft_body.network import MLP
from implicit_soft_body.visualization import render_robot
from implicit_soft_body import IMPLICIT_SOFT_BODY_POLICY_ROOT
import torch
import os
import numpy as np



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
robot = SimpleRobot(device=device)
model_path = os.path.join(IMPLICIT_SOFT_BODY_POLICY_ROOT, "policy_train.pt")
output_path = "output"
os.makedirs(output_path, exist_ok=True)

# Setup parameters for network simlation
min_a = 0.25
max_abs_da = 0.3
center_vertex_id = 8
forward_vertex_id = 6

# x2 = robot.forward(robot.x)
num_epochs = 1
num_frames = 300
loss_history = []
input_history = []
x_history = []
v_history = []
input_size = robot.x.shape[0]
output_size = robot.l0.shape[0]
torch.random.manual_seed(21)
actuation_seq = []
model = MLP(4*input_size, output_size, 24)
model = model.to(device)
model.load_state_dict(torch.load(model_path))

for epoch in range(num_epochs):
    print(f'epoch {epoch}')
    x = robot.x
    v = robot.v
    a = torch.ones_like(robot.l0)
    for i in range(num_frames):
        print(f'frame {i}')
        x = x.to(device)
        v = v.to(device)
        # if i % 20 > 10:
        #     a = torch.ones_like(robot.l0) * 0.3
        # else:
        #     a = torch.ones_like(robot.l0) * 1.0
        # da = model(preprocess(x,v))
        # a = postprocess(a, da)
        # print(a)
        # import pdb; pdb.set_trace()
        # input = preprocess(x, v)
        input = preprocess(x, v, center_id=center_vertex_id, forward_id=forward_vertex_id)
        da = model(input)
        a = postprocess(a, da, noise_std=0.15, max_a=1.0, min_a=min_a, max_abs_da=max_abs_da)
        x, v = robot.forward(x, v, a)
        print("Average x position: ", torch.mean(x[:, 0]))
        input_history.append(input.detach().cpu().numpy())
        actuation_seq.append(a.detach().cpu().numpy())

actuation_seq = np.array(actuation_seq)
input_history = np.array(input_history)
np.save(os.path.join(output_path, "input_history_simulate.npy"), input_history)
np.save(os.path.join(output_path, "actuation_seq_simulate.npy"), actuation_seq)
output = render_robot(actuation_seq, pos=robot.x.detach().cpu().numpy(), triangles=robot.triangles.detach().cpu().numpy(), muscles=robot.springs.detach().cpu().numpy())
with open(os.path.join(output_path, "output.html"), "w") as f:
    f.write(output)




    