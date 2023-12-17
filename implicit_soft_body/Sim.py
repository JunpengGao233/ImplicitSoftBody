from typing import Any
from functools import partial

import torch

import system 
class DiffSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, v0, a, dt, max_iter, robot):
        ctx.robot = robot
        # dx = (1e-3*torch.randn_like(x0)).requires_grad_(True)
        dx = (dt * v0).requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [dx], lr=1e-2, tolerance_change=1e-4, max_iter=5
        )
        # print(dx)
        for epoch_i in range(1000):#range(max_iter):
        # for epoch_i in range(max_iter):
            dx0 = dx.detach().clone()
            a = a.detach().clone().requires_grad_(True)
            x0 = x0.detach().clone()#.requires_grad_(True)
            v0 = v0.detach().clone()#.requires_grad_(True)
            def closure():
                optimizer.zero_grad()
                loss = robot.total_energy(x0, x0+dx, v0, a)
                loss.backward(retain_graph=True)
                # loss.backward()
                return loss
            optimizer.step(closure)
            if torch.norm(dx-dx0)/dx0.norm() < 1e-3:
                break
        print(f'epoch {epoch_i}: loss {dx.norm().item()}, relative loss change {torch.norm(dx-dx0)/dx0.norm()}')
        v = dx/dt
        x = x0+dx
        ctx.save_for_backward(x0, x, v0, v, a)

        return x, v
    
    @staticmethod
    def backward(ctx, grad_output_x, grad_output_v) -> Any:
        print("backward")
        
        x0, x, v0, v, a = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        a = a.detach().clone().requires_grad_(True)
        def energy_partial(x):
            x = x.reshape(-1,2)
            return ctx.robot.total_energy(x0, x, v0, a)
        with torch.enable_grad():
            x_flat = x.flatten()
            E = energy_partial(x_flat)
            f = -(torch.autograd.grad(E, x_flat, create_graph=True)[0])
        dLdx = grad_output_x
        dLdx = dLdx.flatten()
        dfdx = torch.autograd.functional.hessian(energy_partial, x_flat)
        dfdx = torch.transpose(dfdx, 0, 1)
        z = torch.linalg.solve(dfdx, dLdx)

        with torch.enable_grad():
            # f_flat = f.flatten()
            zTf = z.reshape(1,-1) @ f
            dLda = -torch.autograd.grad(zTf, a)[0]
        return None, None, dLda, None, None, None