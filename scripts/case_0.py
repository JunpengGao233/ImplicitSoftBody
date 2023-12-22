"""
Successful Case 0 
"""
import os
import torch
import numpy as np
from implicit_soft_body.robot_model import SimpleRobot
from implicit_soft_body.network import MLP
from implicit_soft_body.utils import preprocess, postprocess
from implicit_soft_body.visualization import render_robot
from implicit_soft_body import IMPLICIT_SOFT_BODY_ROOT

output_path = os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "output")
os.makedirs(output_path, exist_ok=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = 'cpu'
robot = SimpleRobot(device=device)

# Setup parameters for network training
min_a = 0.25
max_abs_da = 0.3
center_vertex_id = 8
forward_vertex_id = 6


# x2 = robot.forward(robot.x)
num_epochs = 30
num_frames = 100
loss_history = []
input_size = robot.x.shape[0]
output_size = robot.l0.shape[0]
torch.random.manual_seed(42)
network = MLP(4*input_size, output_size, 16)
# add pretrain to model
# network = model
network = network.to(device)
# pretrained model
# network.load_state_dict(torch.load("policy.pt"))
optimizer = torch.optim.Adam(network.parameters(),lr=1e-3, maximize=True)
loss_last = 1e10
for epoch in range(num_epochs):
    print(f'epoch {epoch}')
    actuation_seq = []
    def closure():
        optimizer.zero_grad()
        x = robot.x
        v = robot.v
        a = 1 / robot.l0 + 0.15 * torch.randn_like(robot.l0)
        last_p = robot.x_pos(x)
        loss = 0
        for i in range(num_frames):
            x = x.to(device)
            v = v.to(device)
            a = a.to(device)
            da = network(preprocess(x,v, center_id=8, forward_id=6))
            a = postprocess(a, da, noise_std=0.15, max_a=1.0)
            x, v = robot.forward(x, v, a)
            actuation_seq.append(a.detach().cpu().numpy())
            curr_p = robot.x_pos(x)
            loss += curr_p  - last_p
            last_p = curr_p
            
        # loss /= num_frames
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
    if loss >= np.max(loss_history):
        print("saving best, loss:", loss)
        model_dict = network.state_dict()
        torch.save(model_dict, os.path.join(output_path, "policy_train.pt"))
        np.save(os.path.join(output_path, "actuation_seq_best.npy"), actuation_seq)
    # print(actuation_seq)
model_dict = network.state_dict()
loss_history = np.array(loss_history)
np.save(os.path.join(output_path, "loss_history.npy"), loss_history)
torch.save(model_dict, os.path.join(output_path, "policy_train_last.pt"))
actuation_seq = np.array(actuation_seq)
np.save(os.path.join(output_path, "actuation_seq_last.npy"), actuation_seq)




dataset_dir = os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "assets", "dataset")
output_dir = os.path.join(IMPLICIT_SOFT_BODY_ROOT, "..", "output")



robot = SimpleRobot(device='cpu')

pos = robot.x.detach().cpu().numpy()
triangles = robot.triangles.detach().cpu().numpy()
muscles = robot.springs.detach().cpu().numpy()

actuation_seq_path = os.path.join(output_dir, "actuation_seq_best.npy")

best_action = np.load(actuation_seq_path)
output = render_robot(best_action, pos=pos, triangles=triangles, muscles=muscles)
with open(os.path.join(output_dir, "index.html"), "w") as f:
    f.write(output)

