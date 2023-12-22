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
        # self.x = torch.tensor([[1.6825000047683716, 0.5799999833106995], [1.6675000190734863, 0.9549999833106995], [1.315000057220459, 0.9399999976158142], [1.472499966621399, 0.7749999761581421], [1.2100000381469727, 0.4449999928474426], [1.375, 0.32499998807907104], [1.4800000190734863, 1.1649999618530273], [1.225000023841858, 1.4500000476837158], [1.75, 1.4199999570846558], [1.600000023841858, 1.8700000047683716], [2.5, 2.049999952316284], [2.424999952316284, 1.5850000381469727], [2.005000114440918, 0.8949999809265137], [2.755000114440918, 1.5700000524520874], [3.0850000381469727, 1.4950000047683716], [2.7325000762939453, 0.9850000143051147], [2.4024999141693115, 0.925000011920929], [3.077500104904175, 0.9700000286102295], [3.047499895095825, 0.6549999713897705], [2.882499933242798, 0.5950000286102295], [1.5700000524520874, 0.2199999988079071], [1.1349999904632568, 0.009999999776482582], [3.0325000286102295, 0.2800000011920929], [3.302500009536743, 0.29499998688697815], [2.8524999618530273, 0.02500000037252903], [3.497499942779541, 0.12999999523162842], [1.7350000143051147, 0.009999999776482582], [2.0950000286102295, 1.6150000095367432]])
        # self.triangles = torch.tensor([[1, 2, 3], [1, 3, 0], [3, 5, 4], [3, 0, 5], [3, 2, 4], [6, 7, 2], [1, 6, 2], [8, 9, 6], [9, 6, 7], [8, 1, 6], [12, 8, 1], [10, 11, 13], [14, 10, 13], [15, 14, 13], [16, 15, 11], [15, 11, 13], [17, 14, 15], [18, 19, 15], [18, 15, 17], [20, 21, 5], [21, 5, 4], [22, 23, 18], [19, 22, 18], [17, 23, 18], [24, 22, 23], [5, 0, 20], [25, 23, 24], [22, 24, 19], [26, 20, 21], [27, 8, 12], [27, 11, 16], [27, 12, 16], [9, 27, 8], [27, 11, 10], [9, 10, 27]])
        # self.springs = torch.tensor([[2, 4], [8, 1], [1, 12], [12, 8], [15, 11], [15, 16], [16, 11], [18, 17], [18, 23], [17, 23], [0, 20], [5, 0], [5, 20], [22, 19], [22, 24], [19, 24], [12, 16], [3, 2], [4, 3]])
        # self.l0 = torch.tensor([0.5060138702392578, 0.47226181626319885, 0.34279194474220276, 0.5836523175239563, 0.6742079257965088, 0.33541035652160645, 0.6603834629058838, 0.3164254426956177, 0.4411633014678955, 0.7115125060081482, 0.3771687150001526, 0.39947620034217834, 0.2214723825454712, 0.34889116883277893, 0.31212979555130005, 0.5707889795303345, 0.3986302614212036, 0.22810354828834534, 0.42167073488235474])
        self.x = torch.tensor( [
          [0, 0],
          [2, 0],
          [1, 1]
        ], dtype=torch.float32)
        self.triangles = torch.tensor([[0,1,2
                                ]])
        self.springs = torch.tensor([[0,2],[1,2]])
        # self.l0 = torch.tensor([np.sqrt(2), np.sqrt(2)], dtype=torch.float32)
        self.l0 = torch.ones_like(self.x)
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


    # x2 = robot.forward(robot.x)
    num_epochs = 100
    num_frames = 20#00
    loss_history = []
    input_size = robot.x.shape[0]
    output_size = robot.l0.shape[0]
    torch.random.manual_seed(42)
    da = (torch.randn_like(robot.l0) * 0.5).requires_grad_()
    optimizer = torch.optim.Adam([da],lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_last = 1e10
    for epoch in range(num_epochs):
        actuation_seq = []
        optimizer.zero_grad()
        x = robot.x
        v = robot.v
        v = torch.zeros_like(robot.v)
        # v[:,0] = 1
        a = torch.ones_like(robot.l0).requires_grad_(True)
        last_p = robot.x_pos(x)
        # last_p = x
        loss = 0
        for i in range(num_frames):
            x = x.to(device)
            v = v.to(device)
            a = a.to(device)
            a = postprocess(a, da)
            x, v = robot.forward(x, v, a)
            actuation_seq.append(a.detach().cpu().numpy())
            # curr_p = robot.x_pos(x)
            # loss += curr_p - last_p
            # last_p = curr_p
        # loss = -(robot.x_pos(x) - last_p)  *1e3
        # loss = (x - last_p)[:, 0].mean()
        # a_row = a.shape[0]
        # loss = (x[:a_row] - last_p[:a_row]).sum()
        loss = ((x[2] - torch.tensor([1.5,0.5])).pow(2).sum() + (x[0]- torch.tensor([0.0, 0.0])).pow(2).sum() + (x[1]- torch.tensor([2.0, 0.0])).pow(2).sum())*1e3
        print("loss\t", loss, "x\t", x)
        # loss = (x[2]).pow(2).sum()
        # print(da)
        # loss /= num_frames
        # loss *= -1
        # import pdb; pdb.set_trace()

        loss.backward()
        # print("da gradd", da.grad)
        # print("da_before", da)
        optimizer.step()
        scheduler.step()
        # print("da_after", da)
        # print("x", x)
        # print("da", da)
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
            np.save('actuation_seq_best.npy', actuation_seq)
        actuation_seq = np.array(actuation_seq)
        output = render_robot(actions=actuation_seq,pos=robot.x.detach().cpu().numpy(), triangles=robot.triangles.detach().numpy(), muscles=robot.springs.detach().numpy() )
        with open(f'debug{epoch}.html', 'w') as f:
            f.write(output)
        # print(actuation_seq)
    np.save('actuation_seq.npy', actuation_seq) 

