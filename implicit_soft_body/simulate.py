from robot_model import SpringRobot
from load_weights import postprocess, preprocess, model
from visualization import render_robot
import torch
import numpy as np


if __name__ == '__main__':
    robot = SpringRobot()

    # x2 = robot.forward(robot.x)
    num_epochs = 1
    num_frames = 20
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    torch.random.manual_seed(21)
    actuation_seq = []
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        x = robot.x
        v = robot.v
        a = torch.ones_like(robot.l0)
        for i in range(num_frames):
            print(f'frame {i}')
            da = model(preprocess(x,v))
            a = postprocess(a, da)
            # print(a)
            # import pdb; pdb.set_trace()
            x, v = robot.forward(x, v, a)
            actuation_seq.append(a.detach().numpy())
    actuation_seq = np.array(actuation_seq)
    output = render_robot(actuation_seq)
    with open('output.html', 'w') as f:
        f.write(output)


    
    