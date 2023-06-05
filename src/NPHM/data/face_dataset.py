import random
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import trimesh
from typing import Literal
import traceback

from .utils import  uniform_ball
import NPHM.env_paths as env_paths
from NPHM.data.manager import DataManager

NUM_SPLITS = env_paths.NUM_SPLITS
NUM_SPLITS_EXPR = env_paths.NUM_SPLITS_EXPR

ROOT_SURFACE = env_paths.SUPERVISION_IDENTITY
ROOT_DEFORM_OPEN = env_paths.SUPERVISION_DEFORMATION_OPEN


class ScannerData(Dataset):
    def __init__(self,
                 mode : Literal['train', 'val'],
                 n_supervision_points_face : int,
                 n_supervision_points_non_face : int,
                 batch_size : int,
                 sigma_near : float,
                 lm_inds : np.ndarray,
                 is_closed : bool = False):

        self.manager = DataManager()

        self.lm_inds = lm_inds
        self.mode = mode
        self.ROOT = ROOT_SURFACE

        if is_closed:
            self.neutral_expr_index = env_paths.neutrals_closed
            self.neutral_type = 'closed'
        else:
            self.neutral_expr_index = env_paths.neutrals
            self.neutral_type = 'open'


        if mode == 'train':
            self.subjects = self.manager.get_train_subjects(self.neutral_type)
        else:
            self.subjects = self.manager.get_eval_subjects(self.neutral_type)

        # obtain subjects and expression indices used for building batches
        self.subject_steps = []
        for s in self.subjects:
            self.subject_steps += [s]


        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.n_supervision_points_non_face = n_supervision_points_non_face
        print('Dataset has {} subjects'.format(len(self.subject_steps)))
        self.sigma_near = sigma_near

        # pre-fetch g.t. facial anchor points
        if self.lm_inds is not None:
            self.gt_anchors = {}

            for i, iden in enumerate(self.subject_steps):
                self.gt_anchors[iden] = self.manager.get_facial_anchors(subject=iden,
                                                                        expression=self.neutral_expr_index[iden])
        else:
            self.gt_anchors = np.zeros([39, 3])



    def __len__(self):
        return len(self.subject_steps)


    def __getitem__(self, idx):
        iden = self.subject_steps[idx]
        expr = self.neutral_expr_index[iden]

        if self.lm_inds is not None:
            gt_anchors = self.gt_anchors[iden]

        try:
            on_face = np.load(self.manager.get_train_path_identity_face(iden, expr))
            points = on_face[:, :3]
            normals = on_face[:, 3:6]
            non_face = np.load(self.manager.get_train_path_identity_non_face(iden, expr))
            points_outer = non_face[:, :3]
            normals_non_face = non_face[:, 3:6]

            # subsample points for supervision
            sup_idx = np.random.randint(0, points.shape[0], self.n_supervision_points_face)
            sup_points = points[sup_idx, :]
            sup_normals = normals[sup_idx, :]
            sup_idx_non = np.random.randint(0, points_outer.shape[0], self.n_supervision_points_non_face//5)
            sup_points_non = points_outer[sup_idx_non, :]
            sup_normals_non = normals_non_face[sup_idx_non, :]

        except Exception as e:
            print('SUBJECT: {}'.format(iden))
            print('EXPRESSION: {}'.format(expr))
            print(traceback.format_exc())
            return self.__getitem__(np.random.randint(0, self.__len__()))



        # sample points for grad-constraint
        sup_grad_far = uniform_ball(self.n_supervision_points_face // 8, rad=0.5)
        sup_grad_near = np.concatenate([sup_points, sup_points_non], axis=0) + \
                        np.random.randn(sup_points.shape[0]+sup_points_non.shape[0], 3) * self.sigma_near #0.01

        ret_dict = {'points_face': sup_points,
                    'normals_face': sup_normals,
                    'sup_grad_far': sup_grad_far,
                    'sup_grad_near': sup_grad_near,
                    'idx': np.array([idx]),
                    'points_non_face': sup_points_non,
                    'normals_non_face': sup_normals_non,
                    }

        if not self.lm_inds is None:
            ret_dict.update({'gt_anchors': np.array(gt_anchors)})

        return ret_dict

    def get_loader(self, shuffle=True):
        #random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=8, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


class ScannerDeformatioData(Dataset):
    def __init__(self,
                 mode : Literal['train', 'val'],
                 n_supervision_points : int,
                 batch_size : int,
                 lm_inds : np.ndarray
                 ):

        self.manager = DataManager()

        self.neutral_expr_index = env_paths.neutrals

        self.mode = mode

        self.ROOT = ROOT_DEFORM_OPEN
        self.lm_inds = lm_inds


        if mode == 'train':
            self.subjects = self.manager.get_train_subjects(neutral_type='open')
        else:
            self.subjects = self.manager.get_eval_subjects(neutral_type='open')

        print(f'Dataset has  {len(self.subjects)} Identities!')

        self.subject_steps = [] # stores subject id for each data point
        self.steps = [] # stores expression if for each data point
        self.subject_index = [] # defines order of subjects used in training, relevant for auto-decoder

        all_files = []
        for i, s in enumerate(self.subjects):
            expressions = self.manager.get_expressions(s)
            self.subject_steps += len(expressions) * [s, ]
            self.subject_index += len(expressions) * [i, ]
            self.steps += expressions
            all_files.append(expressions)

        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points

        # pre-fetch facial anchors for neutral expression
        self.anchors = {}
        for iden in self.subjects:
            self.anchors[iden] = self.manager.get_facial_anchors(subject=iden, expression=self.neutral_expr_index[iden])


    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):

        expr = self.steps[idx]
        iden = self.subject_steps[idx]
        subj_ind = self.subject_index[idx]

        try:
            point_corresp = np.load(self.manager.get_train_path_deformation(iden, expr))
            valid = np.logical_not( np.any(np.isnan(point_corresp), axis=-1))
            point_corresp = point_corresp[valid, :].astype(np.float32)

        except Exception as e:
            print(iden)
            print(expr)
            print('FAILED')
            return self.__getitem__(0) # avoid crashing of training, dirty


        # subsample points for supervision
        sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points)
        sup_points_neutral = point_corresp[sup_idx, :3]
        sup_points_posed = point_corresp[sup_idx, 3:]

        neutral = sup_points_neutral
        posed = sup_points_posed

        gt_anchors = self.anchors[iden]

        return {'points_neutral': neutral,
                'points_posed': posed,
                'idx': np.array([idx]),
                'iden': np.array([self.subjects.index(iden)]),
                'expr': np.array([expr]),
                'subj_ind': np.array([subj_ind]),
                'gt_anchors': gt_anchors}

    def get_loader(self, shuffle=True):
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=8, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

