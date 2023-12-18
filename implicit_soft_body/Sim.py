from typing import Any
from functools import partial

import torch
import numpy as np 

import system 
class DiffSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, v0, a, dt, max_iter, robot):
        ctx.robot = robot
        dx = (dt * v0).requires_grad_(True)
        with torch.enable_grad():
            optimizer = torch.optim.Adam(
                [dx], lr=1e-2, 
            )
        loss_last = 1e10
        for epoch_i in range(3000):#range(max_iter):
            a = a.detach().clone().requires_grad_(True)
            x0 = x0.detach().clone()#.requires_grad_(True)
            v0 = v0.detach().clone()#.requires_grad_(True)
            def closure():
                optimizer.zero_grad()
                loss = robot.total_energy(x0, x0+dx, v0, a)
                # loss = test_energy(x0+dx, a)
                loss.backward(retain_graph=True)
                # print("grad dx2", dx2.grad)
                # loss.backward()
                return loss
            loss = optimizer.step(closure)
            if torch.norm(loss-loss_last)/loss < 1e-5:
                break
            loss_last = loss
        print(f'epoch {epoch_i}: loss {loss.item()}, dx {torch.norm(dx)}')
        v = dx/dt
        x = x0 + dx
        # x = x0.detach().clone()
        # x[1] = x0[1]+dx
        # v= torch.zeros_like(x)
        ctx.save_for_backward(x0, x, v0, v, a)
        # print("x ", x)
        # print("a", a)
        # print("l0, ", robot.l0)
        # print("error", torch.norm(x[1]-x[0])-a*robot.l0)

        return x, v
    
    @staticmethod
    def backward(ctx, grad_output_x, grad_output_v) -> Any:
        
        x0, x, v0, v, a = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        a = a.detach().clone().requires_grad_(True)
        def energy_partial(x):
            x = x.reshape(-1,2)
            return ctx.robot.total_energy(x0, x, v0, a)
        # def energy_partial(x):
        #     x = x.reshape(-1,2)
        #     return (x-2*a).pow(2).sum()
        # print("x ", x)
        with torch.enable_grad():
            x_flat = x.flatten()
            E = energy_partial(x_flat)
            f = -(torch.autograd.grad(E, x_flat, create_graph=True)[0])
        dLdx = grad_output_x
        # print("grad_output_x ", grad_output_x)
        # print("grad_output_v ", grad_output_v)
        dLdx = dLdx.flatten()
        dfdx = 0 - torch.autograd.functional.hessian(energy_partial, x_flat, create_graph=True)
        # print("eigen of dfdx", torch.linalg.eigvals(dfdx))
        # print("dfdx ", dfdx)
        # print("dLdx ", dLdx)
        # print("dfdx ", dfdx)
        # print("dLdx ", dLdx)
        z = torch.linalg.solve(dfdx, dLdx)
        # z = torch.linalg.lstsq(dfdx, dLdx).solution
        # print("z ", z)
        # print("f ", f)
        # print(f)
        # print(z)

        with torch.enable_grad():
            # f_flat = f.flatten()
            # zTf = z.reshape(1,-1) @ f
            zTf = torch.dot(z, f)
            # print("zTf ", zTf)
        dLda = -torch.autograd.grad(zTf, a)[0]
        # print("dLda ", dLda)
        # dxda = a*np.sqrt(2)/(x-x0)
        # print("dxda", dxda)#a*np.sqrt(2)/(x-x0))
        # print("dLda analytic ", dLdx[2:]@dxda)
        return None, None, dLda, None, None, None