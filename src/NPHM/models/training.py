from __future__ import division
import torch
import torch.optim as optim

import os
from glob import glob
import numpy as np
import math
import wandb
import mcubes
import trimesh
import traceback

from NPHM.models.loss_functions import compute_loss
from NPHM.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from NPHM.models.reconstruction import get_logits
from NPHM import env_paths


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TrainerAutoDecoder(object):

    def __init__(self, decoder, cfg, device, train_dataset, val_dataset, exp_name, project, is_closed=False):
        self.decoder = decoder

        self.latent_codes = torch.nn.Embedding(len(train_dataset), decoder.lat_dim,
                                          max_norm=1.0, sparse=True, device=device).float()
        self.latent_codes_val = torch.nn.Embedding(len(val_dataset), decoder.lat_dim,
                                               max_norm=1.0, sparse=True, device=device).float()

        torch.nn.init.normal_(
            self.latent_codes.weight.data,
            0.0,
            0.1 / math.sqrt(decoder.lat_dim),
            )
        print(self.latent_codes.weight.data.shape)
        print(self.latent_codes.weight.data.norm(dim=-1).mean())
        torch.nn.init.normal_(
            self.latent_codes_val.weight.data,
            0.0,
            0.1 / math.sqrt(decoder.lat_dim),
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

        #TODO
        wandb.init(project=project, entity="YOUR_WANDB_NAME", config=config)

        print('Big Box')
        self.min = [-0.4, -0.6, -0.7]
        self.max = [0.4, 0.6, 0.5]
        #else:
        #    print('Small Box')
        #    self.min = [-0.35, -0.45, -0.15]
        #    self.max = [0.35, 0.35, 0.35]
        self.res = 256
        self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).to(self.device, dtype=torch.float)
        self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3)).to(self.device)

        self.log_steps = 0


        wandb.watch(decoder, log_freq=100)


    def reduce_lr(self, epoch):
        if self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_encoder.param_groups:
                param_group["lr"] = lr

        if epoch > 1000 and self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
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

        loss_dict_nphm = compute_loss(batch, self.decoder, self.latent_codes, self.device)


        loss_tot = 0
        for key in loss_dict_nphm.keys():
            loss_tot += self.cfg['lambdas'][key] * loss_dict_nphm[key]

        loss_tot.backward()


        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

        if self.cfg['grad_clip_lat'] is not None:
            torch.nn.utils.clip_grad_norm_(self.latent_codes.parameters(), max_norm=self.cfg['grad_clip_lat'])
        self.optimizer_encoder.step()
        self.optimizer_lat.step()

        loss_dict = {k: loss_dict_nphm[k].item() for k in loss_dict_nphm.keys()}

        loss_dict.update({'loss': loss_tot.item()})

        return loss_dict


    def train_model(self, epochs):
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

            if epoch % ckp_interval == 0:# and epoch > 0:
                self.save_checkpoint(epoch)
                self.log_recs(epoch)
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
                print_str += " " + k + " {:06.4f} - {:06.4f}".format(sum_loss_dict[k], val_loss_dict[k])
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
        if 'ckpt' in self.cfg and self.cfg['ckpt'] is not None:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.cfg['ckpt'])
        else:
            print('LOADING', checkpoints[-1])
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
            l_dict= compute_loss(val_batch, self.decoder, self.latent_codes_val, self.device)
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


    def log_dict(self, cfg):
        return cfg


    def log_recs(self, epoch):
        self.decoder.eval()
        exp_dir = self.exp_path + 'recs/epoch_{}/'.format(epoch)

        os.makedirs(exp_dir, exist_ok=True)

        num_steps_eval = min(5, len(self.val_dataset)//2)
        num_steps_eval_train = min(5, len(self.val_dataset)//2)
        for jj in range(num_steps_eval_train):
            print('log step', jj)
            try:
                step_train = (jj + num_steps_eval_train * self.log_steps)%len(self.train_dataset)
                step = (jj + num_steps_eval * self.log_steps)%len(self.val_dataset)


                iden_train = self.train_dataset.subject_steps[step_train]
                expr_train = self.train_dataset.neutral_expr_index[iden_train]


                encoding_train = self.latent_codes(
                    torch.from_numpy(np.array([[step_train]])).to(self.device)).squeeze().unsqueeze(0)


                iden = self.val_dataset.subject_steps[step]
                expr = self.val_dataset.neutral_expr_index[iden]

                encoding = self.latent_codes_val(
                    torch.from_numpy(np.array([[step]])).to(self.device)).squeeze().unsqueeze(0)
                print('Reconstruction', step_train, step)

                logits_train = get_logits(self.decoder,
                                          encoding_train,
                                          self.grid_points.clone(),
                                          nbatch_points=25000,
                                          )
                trim_train = mesh_from_logits(logits_train, self.min, self.max, self.res)
                logits_val = get_logits(self.decoder,
                                          encoding,
                                          self.grid_points.clone(),
                                          nbatch_points=25000,
                                          )
                trim_val = mesh_from_logits(logits_val, self.min, self.max, self.res)

                trim_val.export(exp_dir + 'val_{}_{}.ply'.format(iden, expr))
                trim_train.export(exp_dir + 'train_{}_{}.ply'.format(iden_train, expr_train))

            except Exception as e:
                print(traceback.format_exc())

        self.log_steps += 1

        return

