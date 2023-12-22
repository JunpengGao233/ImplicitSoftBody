from typing import Any


import torch

class DiffSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xh, v0, a, dt, max_iter, robot):
        # BDF
        if len(xh) == 2:
            mode = 'bdf'
            x0 = xh[0]
            x1 = xh[1]
            dx = (1/3 *x1 - 1/3*x0 + 2/3*dt*v0).requires_grad_(True)
        else:
            mode = 'euler'
            x0 = xh
            dx = (dt * v0).requires_grad_(True)
        
        ctx.robot = robot
        with torch.enable_grad():
            optimizer = torch.optim.Adam(
                [dx], lr=1e-2, 
            )
        loss_last = 1e10
        for epoch_i in range(max_iter):
            a = a.detach().clone().requires_grad_(True)
            x0 = x0.detach().clone()
            v0 = v0.detach().clone()
            def closure():
                optimizer.zero_grad()
                if mode == 'bdf':
                    loss = robot.total_energy_bdf(x0, x1, x1+dx, v0, a)
                elif mode == 'euler':
                    loss = robot.total_energy_euler(x0, x0+dx, v0, a)
                loss.backward(retain_graph=True)
                return loss
            loss = optimizer.step(closure)
            if torch.norm(loss-loss_last)/loss < 1e-5:
                break
            loss_last = loss
        print(f'epoch {epoch_i}: loss {loss.item()}, dx {torch.norm(dx)}')
        if mode == 'bdf':
            x = x1 + dx
            v = (3*x - 4*x1 + x0)/(2*dt)
        elif mode == 'euler':
            x = x0 + dx
            v = dx/dt
        ctx.save_for_backward(xh, x, v0, v, a)

        return x, v
    
    @staticmethod
    def backward(ctx, grad_output_x, grad_output_v) -> Any:
        # print(grad_output_x)
        
        xh, x, v0, v, a = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        a = a.detach().clone().requires_grad_(True)
        def energy_partial(x):
            x = x.reshape(-1,2)
            if len(xh) == 2:
                x0 = xh[0]
                x1 = xh[1]
                return ctx.robot.total_energy_bdf(x0, x1, x, v0, a)
            elif len(xh) == 1:
                x0 = xh
                return ctx.robot.total_energy_euler(x0, x, v0, a)
        with torch.enable_grad():
            x_flat = x.flatten()
            E = energy_partial(x_flat)
            f = -(torch.autograd.grad(E, x_flat, create_graph=True)[0])
        dLdx = grad_output_x
        dLdx = dLdx.flatten()
        dfdx = 0 - torch.autograd.functional.hessian(energy_partial, x_flat, create_graph=True)
        z = torch.linalg.solve(dfdx, dLdx)

        with torch.enable_grad():
            zTf = torch.dot(z, f)
        dLda = -torch.autograd.grad(zTf, a)[0]
        return None, None, dLda, None, None, None