from typing import Any
from functools import partial

import torch
import numpy as np 

import system 
class DiffSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, v0, a, dt, max_iter, robot):
        ctx.robot = robot
        # dx = (1e-3*torch.randn_like(x0)).requires_grad_(True)
        dx = (dt * v0).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [dx], lr=1e-2, 
        )
        # optimizer = torch.optim.LBFGS(
        #     [dx], lr=1e-2, tolerance_change=1e-4, max_iter=5
        # )
        # dx = torch.randn_like(x0)
        # dx[0] = 0
        # dx.requires_grad_(True)
        # print(dx)
        def test_energy(x, a):
            return (x-2*a).pow(2).sum()
        for epoch_i in range(1000):#range(max_iter):
            dx0 = dx.detach().clone()
            a = a.detach().clone().requires_grad_(True)
            x0 = x0.detach().clone()#.requires_grad_(True)
            v0 = v0.detach().clone()#.requires_grad_(True)
            def closure():
                optimizer.zero_grad()
                # dx = torch.stack([dx1, dx2], dim=0)
                # loss = robot.total_energy(x0, x0+dx, v0, a)
                loss = test_energy(x0+dx, a)
                loss.backward(retain_graph=True)
                # print("grad dx2", dx2.grad)
                # loss.backward()
                return loss
            loss = optimizer.step(closure)
            if torch.norm(dx-dx0)/dx0.norm() < 1e-5:
                break
        print(f'epoch {epoch_i}: loss {loss.item()}, dx {torch.norm(dx)}')
        v = dx/dt
        x = x0+dx
        # v= torch.zeros_like(x)
        ctx.save_for_backward(x0, x, v0, v, a)
        print("x ", x)
        print("a", a)

        return x, v
    
    @staticmethod
    def backward(ctx, grad_output_x, grad_output_v) -> Any:
        
        x0, x, v0, v, a = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        a = a.detach().clone().requires_grad_(True)
        # def energy_partial(x):
        #     x = x.reshape(-1,2)
        #     return ctx.robot.total_energy(x0, x, v0, a)
        def energy_partial(x):
            x = x.reshape(-1,2)
            return (x-2*a).pow(2).sum()
        # print("x ", x)
        with torch.enable_grad():
            x_flat = x.flatten()
            E = energy_partial(x_flat)
            f = -(torch.autograd.grad(E, x_flat, create_graph=True)[0])
        dLdx = grad_output_x
        # print("grad_output_x ", grad_output_x)
        # print("grad_output_v ", grad_output_v)
        dLdx = dLdx.flatten()
        dfdx = -torch.autograd.functional.hessian(energy_partial, x_flat, create_graph=True)
        # print("eigen of dfdx", torch.linalg.eigvals(dfdx))
        # print("dfdx ", dfdx)
        # print("dLdx ", dLdx)
        print("dfdx ", dfdx)
        print("dLdx ", dLdx)
        z = torch.linalg.solve(dfdx, dLdx)
        # z = torch.linalg.lstsq(dfdx, dLdx).solution
        print("z ", z)
        print("f ", f)
        # print(f)
        # print(z)

        with torch.enable_grad():
            # f_flat = f.flatten()
            # zTf = z.reshape(1,-1) @ f
            zTf = torch.dot(z, f)
            print("zTf ", zTf)
        dLda = -torch.autograd.grad(zTf, a)[0]
        print("dLda ", dLda)
        # dxda = a*np.sqrt(2)/(x-x0)
        # print("dxda", dxda)#a*np.sqrt(2)/(x-x0))
        # print("dLda analytic ", dLdx[2:]@dxda)
        return None, None, dLda, None, None, None