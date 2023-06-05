import torch
from NPHM.models.diff_operators import jac, gradient


def broyden(g, x_init, J_inv_init, max_steps=50, cvg_thresh=1e-5, dvg_thresh=1, eps=1e-6):
    """Find roots of the given function g(x) = 0.
    This function is impleneted based on https://github.com/locuslab/deq.
    Tensor shape abbreviation:
        N: number of points
        D: space dimension
    Args:
        g (function): the function of which the roots are to be determined. shape: [N, D, 1]->[N, D, 1]
        x_init (tensor): initial value of the parameters. shape: [N, D, 1]
        J_inv_init (tensor): initial value of the inverse Jacobians. shape: [N, D, D]
        max_steps (int, optional): max number of iterations. Defaults to 50.
        cvg_thresh (float, optional): covergence threshold. Defaults to 1e-5.
        dvg_thresh (float, optional): divergence threshold. Defaults to 1.
        eps (float, optional): a small number added to the denominator to prevent numerical error. Defaults to 1e-6.
    Returns:
        result (tensor): root of the given function. shape: [N, D, 1]
        diff (tensor): corresponding loss. [N]
        valid_ids (tensor): identifiers of converged points. [N]
    """

    # initialization
    x = x_init.clone().detach()
    J_inv = J_inv_init.clone().detach()

    ids_val = torch.ones(x.shape[0]).bool()

    gx = g(x, mask=ids_val)
    update = -J_inv.bmm(gx)

    x_opt = x
    gx_norm_opt = torch.linalg.norm(gx.squeeze(-1), dim=-1)

    delta_gx = torch.zeros_like(gx)
    delta_x = torch.zeros_like(x)

    ids_val = torch.ones_like(gx_norm_opt).bool()

    for _ in range(max_steps):

        # update paramter values
        delta_x[ids_val] = update
        x[ids_val] += delta_x[ids_val]
        delta_gx[ids_val] = g(x, mask=ids_val) - gx[ids_val]
        gx[ids_val] += delta_gx[ids_val]

        # store values with minial loss
        gx_norm = torch.linalg.norm(gx.squeeze(-1), dim=-1)
        ids_opt = gx_norm < gx_norm_opt
        gx_norm_opt[ids_opt] = gx_norm.clone().detach()[ids_opt]
        x_opt[ids_opt] = x.clone().detach()[ids_opt]

        # exclude converged and diverged points from furture iterations
        ids_val = (gx_norm_opt > cvg_thresh) & (gx_norm < dvg_thresh)
        if ids_val.sum() <= 0:
            break

        # compute paramter update for next iter
        vT = (delta_x[ids_val]).transpose(-1, -2).bmm(J_inv[ids_val])
        a = delta_x[ids_val] - J_inv[ids_val].bmm(delta_gx[ids_val])
        b = vT.bmm(delta_gx[ids_val])
        b[b >= 0] += eps
        b[b < 0] -= eps
        u = a / b
        J_inv[ids_val] += u.bmm(vT)
        update = -J_inv[ids_val].bmm(gx[ids_val])

    return {'result': x_opt, 'diff': gx_norm_opt, 'valid_ids': gx_norm_opt < cvg_thresh}



def nabla(decoder_shape, xc, cond, anchors):
    """Get gradients df/dx
    Args:
        xc (tensor): canonical points. shape: [B, N, D]
        cond (dict): conditional input.
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        grad (tensor): gradients. shape: [B, N, D, D]
    """
    xc.requires_grad_(True)

    sdf_pred, _ = decoder_shape(xc, cond, anchors)
    return sdf_pred, gradient(sdf_pred, xc)



def search(obs, cond, decoder_expr, anchors, multi_corresp=True):
    """Search correspondences.
            Args:
                xd (tensor): deformed points in batch. shape: [B, N, D]
                xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
                cond (dict): conditional input.
                tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            Returns:
                xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
                valid_ids (tensor): identifiers of converged points. [B, N, I]
            """
    n_batch, n_point, n_dim = obs.shape
    if multi_corresp:
        num_inits = 5
        xc_init = obs.detach().clone()
        xc_init = xc_init.unsqueeze(2).repeat(1, 1, num_inits, 1)
        offsets = torch.randn(xc_init.shape, device=xc_init.device) * 0.05
        offsets[:, :, 0, :] = 0
        xc_init += offsets
        xc_init = xc_init.reshape(n_batch, n_point*num_inits, 3)

        obs = obs.repeat_interleave(num_inits, dim=1)
        cond = cond[:, 0, :].unsqueeze(1).repeat(1, xc_init.shape[1], 1)
        if anchors is not None:
            anchors = anchors[:, 0, :, :].unsqueeze(1).repeat(1, xc_init.shape[1], 1 ,1)


    else:
        xc_init = obs.detach().clone()


    # compute init jacobians
    J_inv_init = jac(decoder_expr, xc_init, cond, anchors).inverse()


    # reshape init to [?,D,...] for boryden
    xc_init = xc_init.reshape(-1, 3, 1)
    J_inv_init = J_inv_init.flatten(0, 1)

    # construct function for root finding
    def _func(xc_opt, mask=None):
        if multi_corresp:
            xc_opt = xc_opt.reshape(n_batch, -1, 3)
            xd_opt, _ = decoder_expr(xc_opt, cond, anchors) #TODO: do I need mask here?
            xd_opt = xd_opt + xc_opt
        else:
            if cond.shape[0] != 1:
                xd_opt, _ = decoder_expr(xc_opt.reshape(1, xc_opt.shape[0], 3), cond.reshape(1, -1, cond.shape[2]),
                                         None if anchors is None else anchors.reshape(1, -1, anchors.shape[2], 3))  # TODO: do I need mask here?
            else:
                xd_opt, _ = decoder_expr(xc_opt.reshape(1, xc_opt.shape[0], 3), cond, anchors)  # TODO: do I need mask here?
            xd_opt = xd_opt + xc_opt.reshape(1, xc_opt.shape[0], 3)
        if obs.shape[0] != 1:
            error = xd_opt - obs.reshape(1, -1, 3)
        else:
            error = xd_opt - obs
        # reshape to [?,D,1] for boryden
        error = error.flatten(0, 1)[mask].unsqueeze(-1)
        return error

    # run broyden without grad
    with torch.no_grad():
        result = broyden(_func, xc_init, J_inv_init,
                         cvg_thresh=1e-6,
                         dvg_thresh=0.2,
                         max_steps=15)


    # reshape back to [B,N,I,D]
    if multi_corresp:
        xc_opt = result["result"].reshape(n_batch, n_point, -1, 3) #n_batch == 1
        result["valid_ids"] = result["valid_ids"].reshape(n_batch, n_point, num_inits)
    else:
        xc_opt = result["result"].reshape(n_batch, n_point, 3)
        result["valid_ids"] = result["valid_ids"].reshape(n_batch, n_point)


    return xc_opt, result

