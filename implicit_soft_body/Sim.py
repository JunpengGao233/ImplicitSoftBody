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
        a.retain_grad()
        # print(a.requires_grad)
        # print(x.requires_grad)
        # print(a)
        def energy_partial(x):
            return ctx.robot.total_energy(x0, x, v0, a)
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            E = energy_partial(x)
            # E.backward(create_graph=True)
            # f = -x.grad
            f = -(torch.autograd.grad(E, x, create_graph=True, retain_graph=True)[0])
        dLdx = grad_output_x
        dLdx = dLdx.flatten()
        dfdx = torch.autograd.functional.hessian(energy_partial, x)
        dfdx = dfdx.reshape(dLdx.shape[0], dLdx.shape[0])
        z = torch.linalg.solve(dfdx, dLdx)

        # dfda = torch.autograd.grad(f, a)[0]
        f_flat = f.flatten()
        zTf = z @ f_flat
        dfda = torch.autograd.grad(zTf, a)[0]
        # dfda = torch.zeros((f_flat.shape[0], a.shape[0]))
        # dLda = a.grad
        # dLda = torch.autograd.grad(zTf, a, retain_graph=True)[0]
        

        # dfda = torch.autograd.functional.jacobian(f_flat, a)
        # for i in range(dLdx.shape[0]):
        #     dfda[i] = torch.autograd.grad(f_flat[i], a)[0]
        # z = z.reshape(-1, 1)
        # dLda = -z.T @ dfda

        return None, None, dLda, None, None, None