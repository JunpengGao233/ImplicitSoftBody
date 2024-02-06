from typing import Any


import torch

class DiffSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, v0, a, dt, max_iter, robot):
        ctx.robot = robot
        dx = (dt * v0).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [dx], lr=1e-2,
        )
        for epoch_i in range(1000):
            dx0 = dx.detach().clone()
            a = a.detach().clone().requires_grad_(True)
            x0 = x0.detach().clone()
            v0 = v0.detach().clone()
            def closure():
                optimizer.zero_grad()
                loss = robot.total_energy(x0, x0+dx, v0, a)
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
            if torch.norm(dx-dx0)/dx0.norm() < 1e-3:
                break
        v = dx/dt
        x = x0+dx
        ctx.save_for_backward(x0, x, v0, v, a)

        return x, v
    
    @staticmethod
    def backward(ctx, grad_output_x, grad_output_v) -> Any:
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
        dfdx = -torch.autograd.functional.hessian(energy_partial, x_flat)
        z = torch.linalg.solve(dfdx, dLdx)
        with torch.enable_grad():
            zTf = torch.dot(z, f)
            dLda = -torch.autograd.grad(zTf, a)[0]
        return None, None, dLda, None, None, None