import torch
import torch.nn.functional as F
from torch import optim
from pytorch3d.ops import knn_points, knn_gather
import trimesh
import numpy as np
import pyvista as pv
from typing import List, Dict

from NPHM.models.diff_operators import gradient
from NPHM.models.iterative_root_finding import search, jac, nabla


def inference_iterative_root_finding_joint(decoder,
                                           decoder_expr,
                                           all_obs : List[torch.tensor],
                                           lambdas,
                                           n_steps,
                                           schedule_cfg : Dict,
                                           step_scale = 1,
                                           lr_scale=1):

        num_observations = len(all_obs)
        num_observations_per_batch = 5
        num_points_per_observation = 1000
        if hasattr(decoder_expr, 'lat_dim_expr'):
            lat_rep = torch.zeros([num_observations, 1, decoder_expr.lat_dim_expr], device=all_obs[0].device).float()
        else:
            lat_rep = torch.zeros([num_observations, 1, 200], device=all_obs[0].device).float()

        lat_rep.requires_grad = True
        lat_rep_shape = torch.zeros([1, 1, decoder.lat_dim], device = all_obs[0].device)
        lat_rep_shape.requires_grad = True

        opt = optim.Adam(params=[lat_rep_shape], lr=0.01*lr_scale)
        opt_expr = optim.Adam(params=[lat_rep,], lr=0.01*lr_scale)

        for j in range(int(n_steps*step_scale)):
            # eye-balled scheduling of learning rate and weighing of losses
            if int(j/step_scale) in schedule_cfg['lr']:
                for param_group in opt.param_groups:
                    param_group["lr"] /= schedule_cfg['lr'][int(j/step_scale)]
                for param_group in opt_expr.param_groups:
                    param_group["lr"] /= schedule_cfg['lr'][int(j/step_scale)]
            if int(j / step_scale) in schedule_cfg['symm_dist']:
                lambdas['symm_dist'] /= schedule_cfg['symm_dist'][int(j/step_scale)]
            if int(j / step_scale) in schedule_cfg['reg_glob']:
                lambdas['reg_global'] /= schedule_cfg['reg_glob'][int(j/step_scale)]
            if int(j / step_scale) in schedule_cfg['reg_loc']:
                lambdas['reg_loc'] /= schedule_cfg['reg_loc'][int(j / step_scale)]
            if int(j / step_scale) in schedule_cfg['reg_expr']:
                lambdas['reg_expr'] /= schedule_cfg['reg_expr'][int(j / step_scale)]


            opt.zero_grad()
            opt_expr.zero_grad()


            _, anchors = decoder(torch.zeros([1, 1, 3], device=lat_rep.device), lat_rep_shape, None)

            sampled_points = []
            sampled_observations_idx = torch.randint(0, num_observations, [num_observations_per_batch])
            for i in range(num_observations_per_batch):
                sampled_idx = sampled_observations_idx[i]
                n_samps = min(num_points_per_observation, all_obs[sampled_idx].shape[0]) # max number of point in batch
                #n_samps = min(1000, all_obs[i].shape[0]) # max number of point in batch
                subsample_idx = torch.randint(0, all_obs[sampled_idx].shape[0], [n_samps])
                sampled_points.append(all_obs[sampled_idx].clone()[subsample_idx, :])

            obs = torch.stack(sampled_points, dim=0)

            glob_cond = torch.cat([lat_rep_shape.repeat(num_observations_per_batch, 1, 1), lat_rep[sampled_observations_idx.long().cuda(), :, :]], dim=-1)

            n_batch, n_point, n_dim = obs.shape

            # perform iterative root finding in order to obain canonical points corresponding to each observation
            if hasattr(decoder, 'lat_dim_loc'):
                p_corresp, search_result = search(obs,
                                                  glob_cond.repeat(1, obs.shape[1], 1),
                                                  decoder_expr,
                                                  anchors.clone().unsqueeze(1).repeat(num_observations_per_batch, obs.shape[1], 1, 1),
                                                  multi_corresp=False)
            else:
                p_corresp, search_result = search(obs,
                                                  glob_cond.repeat(1, obs.shape[1], 1),
                                                  decoder_expr,
                                                  None,
                                                  multi_corresp=False)

            # do not back-prop through broyden
            p_corresp = p_corresp.detach()

            if anchors is not None:
                _anchors = anchors.clone().unsqueeze(1).repeat(num_observations_per_batch, p_corresp.shape[1], 1, 1)
            else:
                _anchors = None

            # instead of back-prop attach analytic gradient
            preds_posed, _ = decoder_expr(p_corresp, glob_cond.repeat(1, p_corresp.shape[1], 1), _anchors)
            preds_posed += p_corresp
            grad_inv = jac(decoder_expr, p_corresp, glob_cond.repeat(1, p_corresp.shape[1], 1), _anchors).inverse()
            correction = preds_posed - preds_posed.detach()
            correction = torch.einsum("bnij,bnj->bni", -grad_inv.detach(), correction)
            # trick for implicit diff with autodiff:
            # xc = xc_opt + 0 and xc' = correction'
            xc = p_corresp + correction


            # compute loss # TODO don't need case distinction
            if hasattr(decoder, 'lat_dim_loc'):
                sdf, _ = decoder(xc, lat_rep_shape.repeat(num_observations_per_batch, 1, 1), None)
                _, sdf_grad = nabla(decoder, p_corresp, lat_rep_shape.repeat(num_observations_per_batch, 1, 1), None)
            else:
                sdf, _ = decoder(xc, lat_rep_shape.repeat(num_observations_per_batch, xc.shape[1],1), None)
                _, sdf_grad = nabla(decoder, p_corresp, lat_rep_shape.repeat(num_observations_per_batch, xc.shape[1],1), None)


            #if (search_result['valid_ids'].shape) == 2:
            sdf = sdf[search_result['valid_ids'], :]
            #else:
            #    sdf = sdf[:, search_result['valid_ids'], :]
            l = sdf.abs()


            # eye-balled schedule for loss clamping
            l = l[l < 0.1]
            if j > int(250*step_scale):
                l = l[l < 0.05]
            #if j > int(500*step_scale):
            #    l = l[l < 0.025]
            if j > int(500 * step_scale):
                l = l[l < 0.0075]


            loss_dict = {}
            loss_dict['surface'] = l.mean()
            loss_dict['reg_expr'] = (torch.norm(lat_rep[sampled_observations_idx.long().cuda(), :, :], dim=-1) ** 2).mean()

            # TODO refactor
            if hasattr(decoder, 'lat_dim_glob'):
                loss_dict['reg_loc'] = (torch.norm(lat_rep_shape[..., 64:], dim=-1) ** 2).mean()
                loss_dict['reg_global'] = (torch.norm(lat_rep_shape[..., :64], dim=-1) ** 2).mean()

                #unobserved_idx = [26, 27, 28, 29, 30, 31, 20, 21, 39]
                unobserved_idx = [30, 31, 39]
                loss_dict['reg_unobserved'] = 0
                for idx in unobserved_idx:
                    loss_dict['reg_unobserved'] += torch.norm(lat_rep_shape[..., 64+idx*32:64+(idx+1)*32], dim=-1).square().mean()

                loc_lats_symm = lat_rep_shape[:, :,
                                decoder.lat_dim_glob:decoder.lat_dim_glob + 2 * decoder.num_symm_pairs * decoder.lat_dim_loc].view(
                    lat_rep_shape.shape[0], decoder.num_symm_pairs * 2, decoder.lat_dim_loc)

                symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
                loss_dict['symm_dist'] = symm_dist
            else:
                loss_dict['symm_dist'] = 0
                loss_dict['reg_unobserved'] = 0
                loss_dict['reg_loc'] = 0
                loss_dict['reg_global'] = (torch.norm(lat_rep_shape, dim=-1) ** 2).mean()



            loss = 0
            for k in lambdas.keys():
                loss += loss_dict[k] * lambdas[k]
            loss.backward()
            opt.step()
            opt_expr.step()
            if j % 1 == 0:
                print_str = "Epoch: {:5d}".format(j)
                for k in lambdas.keys():
                    print_str += " " + k + " {:02.8f} ".format(loss_dict[k])
                print(print_str, search_result['valid_ids'].sum().item())


        return lat_rep, lat_rep_shape, anchors


def inference_identity_space(decoder,
                             all_obs: List[torch.tensor],
                             lambdas,
                             n_steps,
                             schedule_cfg: Dict,
                             step_scale=1,
                             lr_scale=1):
    num_observations = len(all_obs)
    num_observations_per_batch = 5
    num_points_per_observation = 1000


    lat_rep_shape = torch.zeros([1, 1, decoder.lat_dim], device=all_obs[0].device)
    lat_rep_shape.requires_grad = True

    opt = optim.Adam(params=[lat_rep_shape], lr=0.01 * lr_scale)

    for j in range(int(n_steps * step_scale)):
        # eye-balled scheduling of learning rate and weighing of losses
        if int(j / step_scale) in schedule_cfg['lr']:
            for param_group in opt.param_groups:
                param_group["lr"] /= schedule_cfg['lr'][int(j / step_scale)]
        if int(j / step_scale) in schedule_cfg['symm_dist']:
            lambdas['symm_dist'] /= schedule_cfg['symm_dist'][int(j / step_scale)]
        if int(j / step_scale) in schedule_cfg['reg_glob']:
            lambdas['reg_global'] /= schedule_cfg['reg_glob'][int(j / step_scale)]
        if int(j / step_scale) in schedule_cfg['reg_loc']:
            lambdas['reg_loc'] /= schedule_cfg['reg_loc'][int(j / step_scale)]

        opt.zero_grad()

        _, anchors = decoder(torch.zeros([1, 1, 3], device=lat_rep_shape.device), lat_rep_shape, None)

        sampled_points = []
        sampled_observations_idx = torch.randint(0, num_observations, [num_observations_per_batch])
        for i in range(num_observations_per_batch):
            sampled_idx = sampled_observations_idx[i]
            n_samps = min(num_points_per_observation, all_obs[sampled_idx].shape[0])  # max number of point in batch
            # n_samps = min(1000, all_obs[i].shape[0]) # max number of point in batch
            subsample_idx = torch.randint(0, all_obs[sampled_idx].shape[0], [n_samps])
            sampled_points.append(all_obs[sampled_idx].clone()[subsample_idx, :])

        obs = torch.stack(sampled_points, dim=0)

        glob_cond = lat_rep_shape.repeat(num_observations_per_batch, 1, 1)

        n_batch, n_point, n_dim = obs.shape

        # perform iterative root finding in order to obain canonical points corresponding to each observation



        # compute loss # TODO don't need case distinction
        if hasattr(decoder, 'lat_dim_loc'):
            sdf, _ = decoder(obs, lat_rep_shape.repeat(num_observations_per_batch, 1, 1), None)
        else:
            sdf, _ = decoder(obs, lat_rep_shape.repeat(num_observations_per_batch, obs.shape[1], 1), None)


        l = sdf.abs()

        # eye-balled schedule for loss clamping
        l = l[l < 0.1]
        if j > int(250 * step_scale):
            l = l[l < 0.05]
        if j > int(500 * step_scale):
            l = l[l < 0.0075]

        loss_dict = {}
        loss_dict['surface'] = l.mean()

        # TODO refactor
        if hasattr(decoder, 'lat_dim_glob'):
            loss_dict['reg_loc'] = (torch.norm(lat_rep_shape[..., 64:], dim=-1) ** 2).mean()
            loss_dict['reg_global'] = (torch.norm(lat_rep_shape[..., :64], dim=-1) ** 2).mean()

            # unobserved_idx = [26, 27, 28, 29, 30, 31, 20, 21, 39]
            unobserved_idx = [30, 31, 39]
            loss_dict['reg_unobserved'] = 0
            for idx in unobserved_idx:
                loss_dict['reg_unobserved'] += torch.norm(lat_rep_shape[..., 64 + idx * 32:64 + (idx + 1) * 32],
                                                          dim=-1).square().mean()

            loc_lats_symm = lat_rep_shape[:, :,
                            decoder.lat_dim_glob:decoder.lat_dim_glob + 2 * decoder.num_symm_pairs * decoder.lat_dim_loc].view(
                lat_rep_shape.shape[0], decoder.num_symm_pairs * 2, decoder.lat_dim_loc)

            symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
            loss_dict['symm_dist'] = symm_dist
        else:
            loss_dict['symm_dist'] = 0
            loss_dict['reg_unobserved'] = 0
            loss_dict['reg_loc'] = 0
            loss_dict['reg_global'] = (torch.norm(lat_rep_shape, dim=-1) ** 2).mean()

        loss = 0
        for k in lambdas.keys():
            loss += loss_dict[k] * lambdas[k]
        loss.backward()
        opt.step()
        if j % 1 == 0:
            print_str = "Epoch: {:5d}".format(j)
            for k in lambdas.keys():
                print_str += " " + k + " {:02.8f} ".format(loss_dict[k])

    return lat_rep_shape, anchors



