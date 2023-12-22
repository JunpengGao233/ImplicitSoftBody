import numpy as np
from implicit_soft_body.visualization import render_robot
from implicit_soft_body.robot_model import SimpleRobot
from implicit_soft_body import IMPLICIT_SOFT_BODY_ROOT
import os

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
# with open('output.html', 'w') as f:
#     f.write(output)

# write new mesh to json
# json_dict['pos'] = pos.tolist()
# with open("normalized_mesh_0.json", 'w') as f:
#     json.dump(json_dict, f)
