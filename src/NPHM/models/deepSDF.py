import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class DeepSDF(nn.Module):
    def __init__(
            self,
            lat_dim,
            hidden_dim,
            nlayers=8,
            geometric_init=True,
            radius_init=1,
            beta=100,
            out_dim=1,
            num_freq_bands=None,
            input_dim=3,
    ):
        super().__init__()
        if num_freq_bands is None:
            d_in_spatial = input_dim
        else:
            d_in_spatial = input_dim*(2*num_freq_bands+1)
        d_in = lat_dim + d_in_spatial
        self.lat_dim = lat_dim
        self.input_dim = input_dim
        print(d_in)
        print(hidden_dim)
        dims = [hidden_dim] * nlayers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [nlayers//2]
        self.num_freq_bands = num_freq_bands
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, xyz, lat_rep, anchors=None):

        if self.num_freq_bands is not None:
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz* freq))
                pos_embeds.append(torch.cos(xyz * freq))

            pos_embed = torch.cat(pos_embeds, dim=-1)
            inp = torch.cat([pos_embed, lat_rep], dim=-1)
        else:
            inp = torch.cat([xyz, lat_rep], dim=-1)
        x = inp

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x, None


def sample_point_feature(q, p, fea, var=0.1**2, background=False):
    # q: B x M x 3
    # p: B x N x 3
    # fea: B x N x c_dim
    # p, fea = c

    #print(q.shape)
    #print(p.shape)
    #print(fea.shape)
    # distance betweeen each query point to the point cloud
    dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2
    if background:
        dist_const = torch.ones_like(dist[:, :, :1])* (-0.2)#(-0.025) #hair 0.2
        dist = torch.cat([dist, dist_const], dim=-1)

    weight = (dist / var).exp()  # Guassian kernel

    # weight normalization
    weight = weight / (weight.sum(dim=2).unsqueeze(-1) + 1e-6)
    #print(weight.shape)
    #print(fea.shape)
    #c_out = weight @ fea  # B x M x c_dim
    c_out = (weight.unsqueeze(-1) * fea).sum(dim=2)
    return c_out


class DeformationNetwork(nn.Module):
    def __init__(
            self,
            mode,
            lat_dim_expr,
            lat_dim_id,
            lat_dim_glob_shape,
            lat_dim_loc_shape,
            n_loc,
            anchors,
            hidden_dim,
            nlayers=8,
            out_dim=1,
            input_dim=3,
    ):
        super().__init__()
        self.mode = mode
        self.lat_dim_glob_shape = lat_dim_glob_shape
        self.lat_dim_loc_shape = lat_dim_loc_shape
        self.lat_dim_expr = lat_dim_expr

        self.input_dim = input_dim

        hidden_dim = hidden_dim
        self.num_kps = n_loc
        self.out_dim = out_dim + 1


        # main tried 'glob_only', 'expr_only', 'compress'
        if self.mode == 'glob_only': # F_ex only conditioned on z_ex and z_id_glob
            self.lat_dim = lat_dim_glob_shape + lat_dim_expr
        elif self.mode == 'expr_only': # F_ex only conditioned on z_ex
            self.lat_dim = lat_dim_expr
        elif self.mode == 'interpolate':
            self.lat_dim = lat_dim_glob_shape + lat_dim_expr + lat_dim_loc_shape
        elif self.mode == 'compress': # F_ex conditioned on z_ex and a projection of z_id and anchors to lower dimensions
            self.lat_dim = lat_dim_expr + lat_dim_id
            self.compressor = nn.Sequential(
            nn.Linear((lat_dim_loc_shape + 3) * (n_loc) + lat_dim_loc_shape + lat_dim_glob_shape, 32))

        elif self.mode == 'GNN':
            self.lat_dim = lat_dim_expr * 2

            self.pos_enc = nn.Sequential(nn.Linear(3, lat_dim_loc_shape), nn.ReLU(), nn.Linear(lat_dim_loc_shape, lat_dim_loc_shape))
            self.local_combiner = nn.Sequential(nn.Linear(lat_dim_loc_shape, lat_dim_loc_shape), nn.ReLU(), nn.Linear(lat_dim_loc_shape, lat_dim_loc_shape))
            self.global_combiner = nn.Sequential(nn.Linear(self.lat_dim_glob_shape + (n_loc)*lat_dim_loc_shape, 512),
                                                 nn.ReLU(),
                                                 nn.Linear(512, lat_dim_expr))

        else:
            raise ValueError('Unknown mode!')


        print('creating DeepSDF with...')
        print('lat dim', self.lat_dim)
        print('hidden_dim', hidden_dim)
        self.defDeepSDF = DeepSDF(lat_dim=self.lat_dim,
                                  hidden_dim=hidden_dim,
                                  nlayers=nlayers,
                                  geometric_init=False,
                                  out_dim=out_dim,
                                  input_dim=input_dim).float()

        self.anchors = anchors


    def forward(self,
                xyz : torch.tensor,
                lat_rep : torch.tensor,
                anchors : Optional[torch.tensor]) -> (torch.tensor, torch.tensor):
        '''
         xyz: B x N x 3 : queried 3D coordinates
         lat: B x N x lat_dim : latent code, concatenation of [z_id, z_ex]
         anchors: B x N x n_kps x 3 : facial anchor positions in case F_id uses such

         returns: offsets that model the deformation for each queried points.
           Remaining features are returned separately if there are any
        '''


        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        B, N, _ = xyz.shape

        if self.mode == 'glob_only':
            cond = torch.cat([lat_rep[:, :, :self.lat_dim_glob_shape], lat_rep[..., -self.lat_dim_expr:]], dim=-1)

        elif self.mode == 'expr_only':
            cond = lat_rep[..., -self.lat_dim_expr:]
        elif self.mode == 'interpolate':
            loc_shape_lat = lat_rep[:, 0, self.lat_dim_glob_shape:-self.lat_dim_expr-self.lat_dim_loc_shape].view(B, self.num_kps, self.lat_dim_loc_shape)
            loc_shape_interp = sample_point_feature(xyz[..., :3], anchors[:, 0, :, :3], loc_shape_lat.unsqueeze(1), background=False)
            cond = torch.cat([lat_rep[:, :, :self.lat_dim_glob_shape], loc_shape_interp, lat_rep[..., -self.lat_dim_expr:]], dim=-1)
        elif self.mode == 'compress':
            if not anchors.shape[1] == N:
                if len(anchors.shape) != 4:
                    anchors = anchors.unsqueeze(1).repeat(1, N, 1, 1)
                else:
                    anchors = anchors[:, 0, :, :].unsqueeze(1).repeat(1, N, 1, 1)
            concat = torch.cat([lat_rep[..., :-self.lat_dim_expr], anchors.reshape(B, N, -1)], dim=-1)
            compressed = self.compressor(concat[:, 0, :]).unsqueeze(1).repeat(1, N, 1)
            if self.training: # not exactly sure if this helps
                compressed += torch.randn(compressed.shape, device=compressed.device) / 200

            cond = torch.cat([compressed, lat_rep[..., -self.lat_dim_expr:]], dim=-1)


        elif self.mode == 'GNN':
            positional_offsets = self.pos_enc(anchors[:, 0, :, :])
            local_combined = self.local_combiner(positional_offsets +
                                                 lat_rep[:, 0, self.lat_dim_glob_shape:self.lat_dim_glob_shape+self.num_kps*self.lat_dim_loc_shape].view(B, self.num_kps, 32))
            concat = torch.cat([lat_rep[:, 0, :self.lat_dim_glob_shape], local_combined.view(B, -1)], dim=-1)
            combined = self.global_combiner(concat).unsqueeze(1).repeat(1, N, 1)
            tmp = lat_rep[:, :, -self.lat_dim_expr:]
            cond = torch.cat([combined, tmp], dim=-1)
        else:
            raise ValueError('Unknown mode')

        pred = self.defDeepSDF(xyz, cond)[0]

        return pred[..., :3], pred[..., -1:]

