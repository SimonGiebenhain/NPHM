import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional


class EnsembledLinear(nn.Module):
    '''
    Simple implementation of an "ensembled" linear layer.
    It applies ensemble_size linear layers at once using batched matrix multiplcation
    Also it supports weight sharing between symmetric ensemble members
    '''
    def __init__(self, ensemble_size, n_symm, in_features, out_features, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_symm = n_symm
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(ensemble_size - self.n_symm, out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(ensemble_size - self.n_symm, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # perform usual weight init for each ensemble member individually
        for e in range(self.ensemble_size - self.n_symm):
            torch.nn.init.kaiming_uniform_(self.weight[e, ...], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[e, ...])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[e, ...], -bound, bound)

    def forward(self, input):
        # input: A x B X D_in
        # A: ensemble dimensions
        # B: batch dimension

        # repeat symmetric part of ensamble
        W = torch.cat([
            self.weight[:self.n_symm, ...].repeat_interleave(2, dim=0), self.weight[self.n_symm:, ...]],
                      dim=0)

        # perform matrix multiplication
        output = torch.bmm(W, input.permute(0, 2, 1)).permute(0, 2, 1) # A x B x D_out

        if self.bias is not None:
            b = torch.cat(
                [self.bias[:self.n_symm, ...].repeat_interleave(2, dim=0), self.bias[self.n_symm:, ...]],
                    dim=0)
            output += b.unsqueeze(1)
        return output


class EnsembledDeepSDF(nn.Module):
    '''
    Execute multiple DeepSDF networks in parallel
    '''
    def __init__(
            self,
            ensemble_size,
            n_symm,
            lat_dim,
            hidden_dim,
            nlayers,
            out_dim=1,
            input_dim=3,
    ):
        super().__init__()
        d_in = input_dim+lat_dim

        self.ensemble_size = ensemble_size
        self.n_symm = n_symm
        self.lat_dim = lat_dim
        self.input_dim = input_dim

        dims = [hidden_dim] * nlayers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [nlayers//2]

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
                in_dim = dims[layer]
            else:
                out_dim = dims[layer + 1]
                in_dim = dims[layer]

            lin = EnsembledLinear(self.ensemble_size, self.n_symm, in_dim, out_dim)
            setattr(self, "lin" + str(layer), lin)

        #self.activation = nn.ReLU()
        self.activation = nn.Softplus(beta=100)

    def forward(self, xyz, lat_rep):
        # xyz: A x B x nPoints x 3
        # lat_rep: A x B x nPoints x nFeats
        A, B, nP, _ = xyz.shape

        inp = torch.cat([xyz, lat_rep], dim=-1)

        # merge batch and point dimension
        inp = inp.reshape(A, B*nP, -1) # A x (B*nP) x (3+nFeats)
        x = inp

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        # un-merge dimensions
        x = x.reshape(A, B, nP, -1)

        return x


def sample_point_feature(q, p, fea, var=0.1**2, background=False):
    # q: B x n_points x 3
    # p: B x n_kps x 3
    # fea: B x n_points x n_kps x channel_dim


    # distance betweeen each query point to the point cloud
    dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)
              ).norm(dim=3) + 10e-6) ** 2 # B x n_points, n_kps

    # add "background" mlp that doesn't have anchor point
    if background:
        dist_const = torch.ones_like(dist[:, :, :1])* (-0.2)
        dist = torch.cat([dist, dist_const], dim=-1)

    weight = (dist / var).exp()  # Guassian kernel

    # weight normalization
    weight = weight / (weight.sum(dim=2).unsqueeze(-1) + 1e-6)

    c_out = (weight.unsqueeze(-1) * fea).sum(dim=2) # B x n_points x channel_dim
    return c_out


class FastEnsembleDeepSDFMirrored(nn.Module):
    def __init__(
            self,
            lat_dim_glob : int,
            lat_dim_loc : int,
            n_loc : int, # number of facial anchor points
            n_symm_pairs: int, # number of symmetric pairs of anchors, expected to be listed before non-symmetric
            anchors : torch.tensor, # average anchor positions
            hidden_dim : int,
            n_layers : int,
            pos_mlp_dim : int=256, # hidden dim of MLP_pos
            out_dim : int=1, # dimensionality of the modeled neural field
            input_dim : int=3, # (input) domain of the modeled neural field
    ):
        super().__init__()

        self.lat_dim_glob = lat_dim_glob
        self.lat_dim_loc = lat_dim_loc
        self.lat_dim = lat_dim_glob + (n_loc+1) * lat_dim_loc
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.pos_mlp_dim = pos_mlp_dim

        self.num_kps = n_loc
        self.num_symm_pairs = n_symm_pairs

        lat_dim_part = lat_dim_glob + self.lat_dim_loc
        hidden_dim = hidden_dim

        self.ensembled_deep_sdf = EnsembledDeepSDF(ensemble_size=self.num_kps+1,
                                                   n_symm=self.num_symm_pairs,
                                                   lat_dim=lat_dim_part,
                                                   hidden_dim=hidden_dim,
                                                   nlayers=n_layers,
                                                   out_dim=out_dim,
                                                   input_dim=input_dim,
                                                   ).float()

        self.anchors = anchors

        self.mlp_pos = nn.Sequential(
            nn.Linear(self.lat_dim_glob, self.pos_mlp_dim),
            nn.ReLU(),
            nn.Linear(self.pos_mlp_dim, self.pos_mlp_dim),
            nn.ReLU(),
            nn.Linear(self.pos_mlp_dim, self.num_kps * 3)
        )


    def forward(self,
                xyz :torch.tensor,
                lat_rep : torch.tensor,
                anchors_gt : Optional[torch.tensor]) ->(torch.tensor, torch.tensor):
        '''
        xyz: B x N x 3 : queried 3D coordinates
        lat_rep: B x N x self.lat_dim
        lat_rep is sturctured as follows (!!!):
        first self.lat_dim_glob elements are global latent
        lat_rep = [z_glob, z_1, z*_1, z_2, z*_2, ..., z_{n_symm}, z*_{n_symm}, z_{non_symm_1}, z_{non_symm_2}, ... ]
        anchors_gt is not used!!

        returns: predictd sdf values, and predicted facial anchor positions
        '''

        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        B, N, _ = xyz.shape
        if lat_rep.shape[1] == 1:
            lat_rep = lat_rep.repeat(1, N, 1)

        assert self.lat_dim == lat_rep.shape[-1], 'lat dim {}, lat_rep {}'.format(self.lat_dim, lat_rep.shape)

        # predict anchor positions as offsets to average anchors
        anchors = self.mlp_pos(lat_rep[:, 0, :self.lat_dim_glob]).view(B, self.num_kps, 3) # B x n_kps x 3
        anchors += self.anchors.squeeze(0)


        if len(anchors.shape) < 4:
            anchors = anchors.unsqueeze(1).repeat(1, N, 1, 1) # B x N x n_kps x 3
        else:
            anchors = anchors.repeat(1, N, 1, 1)


        # represent xyz in all local coordinate systems
        # for the very last anchor there is no local coordinate systme, it uses the global one instead
        coords = xyz.unsqueeze(2) - torch.cat([anchors,
                                               torch.zeros_like(anchors[:, :, :1, :])], dim=2)  # B x N x nkps x 3

        # apply mirroring to symmetric anchor pairs
        coords[:, :, 1:2*self.num_symm_pairs:2, 0] *= -1

        # prepare latent codes
        t1 = lat_rep[:, :, :self.lat_dim_glob].unsqueeze(2).repeat(1, 1, self.num_kps+1, 1)
        t2 = lat_rep[:, :, self.lat_dim_glob:].reshape(B, -1, self.num_kps+1, self.lat_dim_loc)
        if t2.shape[1] != N:
            t2 = t2.repeat(1, N, 1, 1)
            t1 = t1.repeat(1, N, 1, 1)
        cond = torch.cat([t1, t2], dim=-1) # B x N x nkps x (dim_glob + dim_loc)

        coords = coords.permute(2, 0, 1, 3) # nkps x B x N x 3
        cond = cond.permute(2, 0, 1, 3) # nkps x B x N x (dim_glob + dim_loc)

        sdf_pred = self.ensembled_deep_sdf(coords, cond)

        # hack, not sure if this is a good idea
        if not self.training:
            sdf_pred[:, :, -1, 0] = 1 #always outside

        sdf_pred = sdf_pred.permute(1, 2, 0, 3) # B x N x nkps x 1
        # blend predictions
        pred = sample_point_feature(xyz[..., :3], anchors[:, 0, :, :3], sdf_pred, background=True, var=0.1**2)

        return pred, anchors[:, 0, :, :]
