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
            dx0 = dx.clone()
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
        
        x0, x, v0, v, a = ctx.saved_tensors
        def energy_partial(x):
            return ctx.robot.total_energy(x0, x, v0, a)
        E = energy_partial(x)
        f = -(torch.autograd.grad(E, x, create_graph=True)[0])
        dLdx = grad_output_x
        dLdx = dLdx.flatten()
        dfdx = torch.autograd.functional.hessian(energy_partial, x)
        dfdx = dfdx.reshape(dLdx.shape[0], dLdx.shape[0])
        z = torch.linalg.solve(dfdx, dLdx)

        f_flat = f.flatten()
        dfda = torch.zeros((f_flat.shape[0], a.shape[0]))
        for i in range(dLdx.shape[0]):
            dfda[i] = torch.autograd.grad(f_flat[i], a, retain_graph=True)[0]
        z = z.reshape(-1, 1)
        dLda = -z.T @ dfda

        return dLda