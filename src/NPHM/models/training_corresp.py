from __future__ import division
import torch
import torch.optim as optim

import os
from glob import glob
import numpy as np
from time import time
import wandb


import traceback
from NPHM.models.loss_functions import compute_loss_corresp_forward
from NPHM.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from NPHM import env_paths
from NPHM.data.manager import DataManager
from NPHM.models.reconstruction import get_logits, deform_mesh



def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TrainerAutoDecoder(object):

    def __init__(self,
                 decoder,
                 decoder_shape,
                 cfg,
                 device,
                 train_dataset,
                 val_dataset,
                 exp_name : str,
                 project : str,
                 ckpt=None):

        self.manager = DataManager()

        self.decoder = decoder.to(device)
        if decoder_shape is not None:
            self.decoder_shape = decoder_shape.to(device)
        else:
            self.decoder_shape = None

        self.ckpt = ckpt
        sub_inds_train = np.unique(np.array(train_dataset.subject_index))
        sub_inds_val = np.unique(np.array(val_dataset.subject_index))
        self.latent_codes_shape = torch.nn.Embedding(sub_inds_train.shape[0], decoder_shape.lat_dim,
                                               max_norm=1.0, sparse=True, device=device).float()
        self.latent_codes_shape_val = torch.nn.Embedding(sub_inds_val.shape[0], decoder_shape.lat_dim,
                                                   max_norm=1.0, sparse=True, device=device).float()
        self.latent_codes_shape.requires_grad = False
        self.latent_codes_shape_val.requires_grad = False

        self.latent_codes = torch.nn.Embedding(len(train_dataset), self.decoder.lat_dim_expr,
                                               max_norm=1.0, sparse=True, device=device).float()
        self.latent_codes_val = torch.nn.Embedding(len(val_dataset), self.decoder.lat_dim_expr,
                                                   max_norm=1.0, sparse=True, device=device).float()


        path_space_space = env_paths.EXPERIMENT_DIR + '/{}/checkpoints/'.format(cfg['training']['shape_exp_name'])
        self.init_shape_state(cfg['training']['shape_ckpt'], path_space_space)

        torch.nn.init.normal_(
            self.latent_codes.weight.data,
            0.0,
            0.01
        )
        print(self.latent_codes.weight.data.shape)
        print(self.latent_codes.weight.data.norm(dim=-1).mean())
        torch.nn.init.normal_(
            self.latent_codes_val.weight.data,
            0.0,
            0.01
        )


        print('Number of Parameters in decoder: {}'.format(count_parameters(self.decoder)))
        self.cfg = cfg['training']
        self.device = device
        self.optimizer_encoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr=self.cfg['lr'],
                                             weight_decay=self.cfg['weight_decay'])
        self.optimizer_lat = optim.SparseAdam(list(self.latent_codes.parameters()), lr=self.cfg['lr_lat'])
        self.optimizer_lat_val = optim.SparseAdam(list(self.latent_codes_val.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = env_paths.EXPERIMENT_DIR + '/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.val_min = None

        self.val_data_loader = self.val_dataset.get_loader()

        config = self.log_dict(cfg)

        # TODO
        wandb.init(project=project, entity="YOUR_WANDB_NAME", config=config)

        self.min = [-0.35, -0.45, -0.15]
        self.max = [0.35, 0.35, 0.35]
        self.res = 256
        self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).to(self.device, dtype=torch.float)
        self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3)).to(self.device)

        self.past_eval_steps = 0
        self.eval_perm = {'train': np.random.permutation(np.arange(len(self.train_dataset))),
                          'val': np.random.permutation(np.arange(len(self.val_dataset)))}

        print('almost done init trainer')
        wandb.watch(decoder, log_freq=100)
        print('Done init trainer')

    def init_shape_state(self, ckpt, path):
        path = path + 'checkpoint_epoch_{}.tar'.format(ckpt)
        checkpoint = torch.load(path)
        self.decoder_shape.load_state_dict(checkpoint['decoder_state_dict'])
        self.latent_codes_shape.load_state_dict(checkpoint['latent_codes_state_dict'])
        print('Train shape space loaded with dims: ')
        print(self.latent_codes_shape.weight.shape)
        self.latent_codes_shape_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
        print('Loaded checkpoint from: {}'.format(path))


    def reduce_lr(self, epoch):
        if epoch > 0 and self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_encoder.param_groups:
                param_group["lr"] = lr

        if epoch > 0 and self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_lat.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_lat_val.param_groups:
                param_group["lr"] = lr


    def train_step(self, batch, epoch):
        self.decoder.train()
        self.optimizer_encoder.zero_grad()
        self.optimizer_lat.zero_grad()

        loss_dict = compute_loss_corresp_forward(batch, self.decoder, self.decoder_shape, self.latent_codes, self.latent_codes_shape, self.device, epoch, self.exp_path)


        loss_tot = 0
        for key in loss_dict.keys():
            loss_tot += self.cfg['lambdas'][key] * loss_dict[key]

        loss_tot.backward()
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])
        if self.cfg['grad_clip_lat'] is not None:
            torch.nn.utils.clip_grad_norm_(self.latent_codes.parameters(), max_norm=self.cfg['grad_clip_lat'])

        self.optimizer_encoder.step()
        self.optimizer_lat.step()
        loss_dict = {k: v.item() for (k, v) in zip(loss_dict.keys(), loss_dict.values())}
        loss_dict.update({'loss': loss_tot.item()})
        return loss_dict


    def train_model(self, epochs):
        print('Starting to train model.')
        loss = 0
        start = self.load_checkpoint()
        ckp_interval = self.cfg['ckpt_interval']

        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss': 0.0})

            train_data_loader = self.train_dataset.get_loader()

            for batch in train_data_loader:
                loss_dict = self.train_step(batch, epoch)
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]
                wandb.log(loss_dict)


            if epoch % ckp_interval == 0:
                self.save_checkpoint(epoch)
                self.log_recs(epoch)
                self.log_recs(epoch, 'train')
            val_loss_dict = self.compute_val_loss(epoch)

            if self.val_min is None:
                self.val_min = val_loss_dict['loss']

            if val_loss_dict['loss'] < self.val_min:
                self.val_min = val_loss_dict['loss']
                for path in glob(self.exp_path + 'val_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss_dict['loss']])

            n_train = len(train_data_loader)

            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train

            print_str = "Epoch: {:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:01.7f} - {:01.7f}".format(sum_loss_dict[k], val_loss_dict[k])
            print(print_str)

            sum_loss_dict.update({'val_' + k: v for (k,v) in zip(val_loss_dict.keys(), val_loss_dict.values())})

            wandb.log(sum_loss_dict)


    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):

            torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_encoder_state_dict': self.optimizer_encoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_lat.state_dict(),
                        'optimizer_lat_val_state_dict': self.optimizer_lat_val.state_dict(),
                        'latent_codes_state_dict': self.latent_codes.state_dict(),
                        'latent_codes_val_state_dict': self.latent_codes_val.state_dict()},
                       path)


    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        if self.ckpt is not None:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.ckpt)
        else:
            if self.cfg['ckpt'] is not None:
                path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.cfg['ckpt'])
            else:
                path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
        self.optimizer_lat.load_state_dict(checkpoint['optimizer_lat_state_dict'])
        self.optimizer_lat_val.load_state_dict(checkpoint['optimizer_lat_val_state_dict'])
        self.latent_codes.load_state_dict(checkpoint['latent_codes_state_dict'])
        self.latent_codes_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
        epoch = checkpoint['epoch']
        for param_group in self.optimizer_encoder.param_groups:
            print('Setting LR to {}'.format(self.cfg['lr']))
            param_group['lr'] = self.cfg['lr']
        for param_group in self.optimizer_lat.param_groups:
            print('Setting LR to {}'.format(self.cfg['lr_lat']))
            param_group['lr'] = self.cfg['lr_lat']
        if self.cfg['lr_decay_interval'] is not None:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_encoder.param_groups:
                param_group["lr"] = self.lr * self.cfg['lr_decay_factor']**decay_steps
        if self.cfg['lr_decay_interval_lat'] is not None:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_lat.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_lat_val.param_groups:
                param_group["lr"] = lr
        return epoch


    def compute_val_loss(self, epoch):
        self.decoder.eval()

        sum_val_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
        sum_val_loss_dict.update({'loss': 0.0})

        c = 0
        for val_batch in self.val_data_loader:

            self.optimizer_lat_val.zero_grad()
            l_dict = compute_loss_corresp_forward(val_batch, self.decoder, self.decoder_shape, self.latent_codes_val,
                                  self.latent_codes_shape_val, self.device, None)

            for k in l_dict.keys():
                sum_val_loss_dict[k] += l_dict[k].item()
            val_loss = 0.0
            for key in l_dict.keys():
                val_loss += self.cfg['lambdas'][key] * l_dict[key]
            val_loss.backward()
            if self.cfg['grad_clip_lat'] is not None:
                torch.nn.utils.clip_grad_norm_(self.latent_codes_val.parameters(), max_norm=self.cfg['grad_clip_lat'])
            self.optimizer_lat_val.step()

            sum_val_loss_dict['loss'] += val_loss.item()
            c = c + 1

        for k in sum_val_loss_dict.keys():
            sum_val_loss_dict[k] /= c
        return sum_val_loss_dict


    def log_dict(self, cfg): # TODO
        return cfg


    def log_recs(self, epoch, mode='val'):
        try:
            d_set = self.val_dataset
            lat_codes = self.latent_codes_val
            if mode == 'train':
                d_set = self.train_dataset
                lat_codes = self.latent_codes
            lat_codes_shape = self.latent_codes_shape_val
            if mode == 'train':
                lat_codes_shape = self.latent_codes_shape
            self.decoder.eval()
            exp_dir = self.exp_path + 'recs/{}_epoch_{}/'.format(mode, epoch)
            os.makedirs(exp_dir, exist_ok=True)
            N_recs = 5
            for jj in range(N_recs):
                rnd_idx = self.eval_perm[mode][(jj + self.past_eval_steps) % len(d_set)]
                self.past_eval_steps += 1

                subj = d_set.subject_steps[rnd_idx]
                expr = d_set.steps[rnd_idx]
                subj_ind = d_set.subject_index[rnd_idx]
                can_expr = d_set.neutral_expr_index[subj]

                encoding_expr = lat_codes(
                    torch.from_numpy(np.array([[rnd_idx]])).to(self.device)).squeeze().unsqueeze(0)

                encoding_shape = lat_codes_shape(torch.from_numpy(np.array([[subj_ind]])).to(self.device)).squeeze().unsqueeze(0)


                m_gt = self.manager.get_registration_mesh(subject=subj,
                                                          expression=can_expr)
                m_gt_posed = self.manager.get_registration_mesh(subject=subj,
                                                          expression=expr)


                if self.decoder_shape.mlp_pos is not None:
                    gt_anchors = self.decoder_shape.mlp_pos(encoding_shape[..., :self.decoder_shape.lat_dim_glob]).view(
                        encoding_expr.shape[0], -1, 3)
                    gt_anchors += self.decoder.anchors.squeeze(0)
                else:
                    gt_anchors = None

                # do Marching cubes and deform results
                trim, trim_deformed = self.construct_rec(encoding_expr,
                                                         encoding_shape,
                                                         None,
                                                         epoch=epoch,
                                                         anchors=gt_anchors)
                # deform registered mesh directly
                trim_reg, trim_deformed_reg = self.construct_rec(encoding_expr,
                                                                 encoding_shape,
                                                                 m_gt,
                                                                 epoch=epoch,
                                                                 anchors=gt_anchors)


                trim.export(exp_dir + 'mesh_{}_neutral.ply'.format(subj))
                m_gt_posed.export(exp_dir + 'gt_{}_e{}.ply'.format(subj, expr))
                trim_deformed.export(exp_dir + 'mesh_{}_e{}.ply'.format(subj, expr))

                trim_reg.export(exp_dir + 'reg_{}_neutral.ply'.format(subj))
                trim_deformed_reg.export(exp_dir + 'reg_{}_e{}.ply'.format(subj, expr))

            print('Done with eval and logged!')
            return
        except Exception as e:
            print(e)
            traceback.print_exc()


    def construct_rec(self, encoding_expr, encoding_shape, mesh, epoch=-1, anchors=None):

        if mesh is None:
            # reconstruct neutral geometry from implicit repr.
            encoding_shape = encoding_shape.unsqueeze(0)
            encoding_expr = encoding_expr.unsqueeze(0)
            logits = get_logits(decoder=self.decoder_shape,
                                encoding=encoding_shape,
                                grid_points=self.grid_points.clone(),
                                nbatch_points=25000)
            mesh = mesh_from_logits(logits, self.min, self.max, self.res)

        deformed_mesh = deform_mesh(mesh, self.decoder, encoding_expr, anchors, lat_rep_shape=encoding_shape)

        return mesh, deformed_mesh





