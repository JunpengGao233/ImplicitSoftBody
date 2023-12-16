from typing import Any
from functools import partial

import torch

import system 
class DiffSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, v0, a, dt, max_iter, robot):
        ctx.robot = robot
        dx = torch.zeros_like(x0).requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [dx], lr=1e-2, tolerance_change=1e-4, max_iter=5
        )

        for epoch_i in range(max_iter):
            dx0 = dx.detach().clone()
            a = a.detach().clone().requires_grad_(True)
            x0 = x0.detach().clone().requires_grad_(True)
            v0 = v0.detach().clone().requires_grad_(True)
            def closure():
                optimizer.zero_grad()
                loss = robot.total_energy(x0, x0+dx, v0, a)
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
            if torch.norm(dx-dx0)/dx0.norm() < 1e-4:
                break
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
            return ctx.robot.total_energy(x0, x, v0, a)
        with torch.enable_grad():
            E = energy_partial(x)
            f = -(torch.autograd.grad(E, x, create_graph=True)[0])
        dLdx = grad_output_x
        dLdx = dLdx.flatten()
        dfdx = torch.autograd.functional.hessian(energy_partial, x)
        dfdx = dfdx.reshape(dLdx.shape[0], dLdx.shape[0])
        z = torch.linalg.solve(dfdx, dLdx)

        with torch.enable_grad():
            f_flat = f.flatten()
            zTf = z @ f_flat
            dLda = -torch.autograd.grad(zTf, a)[0]
        return None, None, dLda, None, None, None