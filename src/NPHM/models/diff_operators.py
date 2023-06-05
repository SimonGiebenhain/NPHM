import torch
from torch.autograd import grad


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status

def jac(decoder_expr, xc, cond, anchors):
    """Get gradients df/dx
    Args:
        xc (tensor): canonical points. shape: [B, N, D]
        cond (dict): conditional input.
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        grad (tensor): gradients. shape: [B, N, D, D]
    """
    xc.requires_grad_(True)

    xd, _ = decoder_expr(xc, cond, anchors)
    xd = xc + xd

    grads = []
    for i in range(xd.shape[-1]):
        d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
        d_out[:, :, i] = 1
        grad = torch.autograd.grad(
            outputs=xd,
            inputs=xc,
            grad_outputs=d_out,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads.append(grad)

    return torch.stack(grads, dim=-2)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(outputs, inputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True)[0][:, :, -3:]
    return points_grad



