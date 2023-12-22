# ImplicitSoftBody

### Package 
|--implicit_soft_body
|  |-- energy: all the energy class defined
|  |-- network: neural network class
|  |--


├── implicit_soft_body
├── energy : all the energy class
├── network.py : neural network class
├── system.py : base robot class
├── robot_model.py : include all the robot models
├── Sim.py : differentiable simulator


### Script Folder

The folder `scripts` contains different scripts using the package.

* `normalize_mesh.py` is to normalize the given mesh to make it located within the boundary of visualization.
* `pretrain.py` uses the given log of input and output of a control policy to imitate this control policy
* `simulate.py` loads a trained model to simulate and output the sequence of actuations for visualization
* `train.py` trains a model to predict the actuation given the position and velocity of nodes of soft robot.
* `visualize_actuation_seq.py`: visualize the motion of soft robot given the sequence of actuation.

### To run the code

1. Install Depencencies (`jinja2`,`numpy`, `Pytorch`)and locally install the package (Tested on Python 3.9)
    `pip install -e .` . In addition, **Pytorch** should be installed.

2.  Run the script `case_0.py` or
    Run the script in the following order: `train.py` -> `simulate.py` -> `visualize_actuation_seq.py`

3. In the `output` folder, you can find the visualized `index.html` file

We provide different training cases. Not all cases are stable. The training on the differentiable simulator is not easy. We did add some noise to the actuation during the training. Training of control policy of simple robots is easier than that of complex one.

