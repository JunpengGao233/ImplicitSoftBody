
from typing import Optional
import numpy as np
import os
from jinja2 import Template, Environment, FileSystemLoader

__CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
__ENV = Environment(loader=FileSystemLoader(__CURRENT_PATH))
__TEMPLATE = __ENV.get_template('template.html')

__POS = np.array([[1.6825000047683716, 0.5799999833106995], [1.6675000190734863, 0.9549999833106995], [1.315000057220459, 0.9399999976158142], [1.472499966621399, 0.7749999761581421], [1.2100000381469727, 0.4449999928474426], [1.375, 0.32499998807907104], [1.4800000190734863, 1.1649999618530273], [1.225000023841858, 1.4500000476837158], [1.75, 1.4199999570846558], [1.600000023841858, 1.8700000047683716], [2.5, 2.049999952316284], [2.424999952316284, 1.5850000381469727], [2.005000114440918, 0.8949999809265137], [2.755000114440918, 1.5700000524520874], [3.0850000381469727, 1.4950000047683716], [2.7325000762939453, 0.9850000143051147], [2.4024999141693115, 0.925000011920929], [3.077500104904175, 0.9700000286102295], [3.047499895095825, 0.6549999713897705], [2.882499933242798, 0.5950000286102295], [1.5700000524520874, 0.2199999988079071], [1.1349999904632568, 0.009999999776482582], [3.0325000286102295, 0.2800000011920929], [3.302500009536743, 0.29499998688697815], [2.8524999618530273, 0.02500000037252903], [3.497499942779541, 0.12999999523162842], [1.7350000143051147, 0.009999999776482582], [2.0950000286102295, 1.6150000095367432]])
__TRIANGLES = np.array([[1, 2, 3], [1, 3, 0], [3, 5, 4], [3, 0, 5], [3, 2, 4], [6, 7, 2], [1, 6, 2], [8, 9, 6], [9, 6, 7], [8, 1, 6], [12, 8, 1], [10, 11, 13], [14, 10, 13], [15, 14, 13], [16, 15, 11], [15, 11, 13], [17, 14, 15], [18, 19, 15], [18, 15, 17], [20, 21, 5], [21, 5, 4], [22, 23, 18], [19, 22, 18], [17, 23, 18], [24, 22, 23], [5, 0, 20], [25, 23, 24], [22, 24, 19], [26, 20, 21], [27, 8, 12], [27, 11, 16], [27, 12, 16], [9, 27, 8], [27, 11, 10], [9, 10, 27]])
__SPRINGS = np.array([[2, 4], [8, 1], [1, 12], [12, 8], [15, 11], [15, 16], [16, 11], [18, 17], [18, 23], [17, 23], [0, 20], [5, 0], [5, 20], [22, 19], [22, 24], [19, 24], [12, 16], [3, 2], [4, 3]])

def render_robot(actions:Optional[np.ndarray]=None,
    pos:np.ndarray=__POS,
    triangles:np.ndarray=__TRIANGLES, 
    muscles:np.ndarray=__SPRINGS, 
    template:Template=__TEMPLATE) -> Optional[str]:
    """
    Args:
        actions (Optional[np.ndarray]): Actions of muscles
        pos (np.ndarray): Position of nodes.
        triangles (np.ndarray): Indices of triangles of soft robots
        muscles (np.ndarray): Indices of muscles of soft robots
        template (jinja2.Template): template to render
    """
    return template.render(pos=pos, triangles=triangles, muscles=muscles, actions=actions)


if __name__=="__main__":
    actions = np.array([0.5060138702392578, 0.47226181626319885, 0.34279194474220276, 0.5836523175239563, 0.6742079257965088, 0.33541035652160645, 0.6603834629058838, 0.3164254426956177, 0.4411633014678955, 0.7115125060081482, 0.3771687150001526, 0.39947620034217834, 0.2214723825454712, 0.34889116883277893, 0.31212979555130005, 0.5707889795303345, 0.3986302614212036, 0.22810354828834534, 0.42167073488235474])
    actions = actions[None, :]
    random_actions = np.random.rand(100, actions.shape[1]) * 0.5 + 0.9
    random_actions = random_actions * actions

    import json
    with open("normalized_mesh_0.json", 'r') as f:
        json_dict = json.load(f)    
    pos = np.array(json_dict['pos'])
    triangles = np.array(json_dict['triangles'])
    muscles = np.array(json_dict['springs'])


    # # Scale robots to the same size
    # max_pos = np.max(pos, axis=0)
    # min_pos = np.min(pos, axis=0)

    # max_pos_default = np.max(__POS, axis=0)
    # min_pos_default = np.min(__POS, axis=0)

    # scale = (max_pos_default - min_pos_default) / (max_pos - min_pos)

    # # translate to the same center
    # center_pos = (max_pos + min_pos) / 2
    # center_pos_default = (max_pos_default + min_pos_default) / 2
    # pos = (pos - center_pos) * scale + center_pos_default


    # output = render_robot(pos=pos, triangles=triangles, muscles=muscles)

    # output = render_robot(random_actions)
    best_action = np.load("actuation_seq_best.npy")
    output = render_robot(best_action, pos=pos, triangles=triangles, muscles=muscles)
    with open('best_trained.html', 'w') as f:
        f.write(output)
    # with open('output.html', 'w') as f:
    #     f.write(output)
    
    # write new mesh to json
    # json_dict['pos'] = pos.tolist()
    # with open("normalized_mesh_0.json", 'w') as f:
    #     json.dump(json_dict, f)

