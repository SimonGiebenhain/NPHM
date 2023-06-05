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


class JointScannerData(Dataset):
    def __init__(self,
                 mode : Literal['train', 'val'],
                 n_sup : int,
                 n_sup_outer : int,
                 n_sup_off : int,
                 n_sup_corresp,
                 batch_size,
                 sigma_near,
                 lm_inds):

        env_paths.neutrals = env_paths.neutrals_closed
        subjects_train = [s for s in env_paths.neutrals if not s in env_paths.subjects_eval and env_paths.neutrals[s] >= 0]

        self.neutral_expr_index = env_paths.neutrals

        print('Dataset has number of iden: ')
        print(len(subjects_train))

        self.mode = mode

        self.ROOT_def = ROOT_DEFORM_CLOSED
        self.ROOT = ROOT_SURFACE
        self.lm_inds = lm_inds


        if mode == 'train':
            self.subjects = [s for s in subjects_train if env_paths.neutrals[s] >= 0]
        else:
            self.subjects = [s for s in env_paths.subjects_eval if env_paths.neutrals[s] >= 0]

        files = []
        for s in self.subjects:
            exprs = [int(x.split('_')[-1]) for x in os.listdir(self.ROOT_regi + s) if x.startswith('expression')]
            for e in exprs:
                files.append(s + '_{:03d}'.format(e))


        self.subject_lens = {}

        self.subject_steps = []
        self.steps = []
        self.subject_index = []
        self.load_options = []
        self.neutral_steps = {}
        cnt = 0

        for i, s in enumerate(self.subjects):
            fs = [f for f in files if s == f.rpartition('_')[0]]

            self.subject_lens[s] = len(fs)
            self.subject_steps += len(fs) * [s, ]
            self.subject_index += len(fs) * [i, ]
            self.steps += [int(f.split('_')[-1]) for f in fs]
            for c, f in enumerate(fs):
                if int(f.split('_')[-1]) == self.neutral_expr_index[s]:
                    self.neutral_steps[s] = cnt + c
            if s not in self.neutral_steps:
                print(fs)
            assert s in self.neutral_steps
            cnt += len(fs)

        self.batch_size = batch_size
        self.n_supervision_points = n_sup
        self.n_supervision_outer = n_sup_outer
        self.n_supervision_off = n_sup_off
        self.n_supervision_corresp = n_sup_corresp
        self.sigma_near = sigma_near

        self.anchors = {}

        for i, iden in enumerate(self.subject_steps):

            mesh_path = self.ROOT_regi + '/{}/expression_{}/warped.ply'.format(iden, self.neutral_expr_index[iden])
            m = trimesh.load(mesh_path, process=False)

            gt_anchors = np.array(m.vertices[self.lm_inds, :]) / 25
            self.anchors[iden] = gt_anchors



    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):

        expr = self.steps[idx]
        iden = self.subject_steps[idx]

        if np.random.rand() < 0.1:
            expr = self.neutral_expr_index[iden]
            idx = self.neutral_steps[iden]


        subj_ind = self.subject_index[idx]

        is_neutral = expr == self.neutral_expr_index[iden]

        ###################################################
        # load correspondences
        ###################################################
        rnd_file = np.random.randint(0, NUM_SPLITS_EXPR)
        point_corresp = np.load(self.ROOT_def + iden + '/expression_{}'.format(expr) + '/corresp_{}.npy'.format(rnd_file))
        valid = np.logical_not( np.any(np.isnan(point_corresp), axis=-1))
        point_corresp = point_corresp[valid, :].astype(np.float32)


        ###################################################
        # load surface points
        ###################################################
        rnd_file = np.random.randint(0, NUM_SPLITS)
        rnd_file_outer = np.random.randint(0, NUM_SPLITS)
        face_data = np.load(
            self.ROOT + iden + '/face_{}_{}.npy'.format(expr, rnd_file) )

        non_face_data = np.load(
            self.ROOT + iden + '/non_face_{}_{}.npy'.format(expr, rnd_file_outer))

        off_surface_data = np.load(
            self.ROOT + iden + '/off_{}_{}.npy'.format(expr, rnd_file_outer))

        neutral_face_data = np.load(
            self.ROOT + iden + '/face_{}_{}.npy'.format(self.neutral_expr_index[iden], rnd_file))

        neutral_non_face_data = np.load(
            self.ROOT + iden + '/non_face_{}_{}.npy'.format(self.neutral_expr_index[iden], rnd_file_outer))

        neutral_off_surface_data = np.load(
            self.ROOT + iden + '/off_{}_{}.npy'.format(self.neutral_expr_index[iden], rnd_file_outer))

        ###################################################
        # subsample points for supervision
        ###################################################
        sup_idx_corresp = np.random.randint(0, point_corresp.shape[0], self.n_supervision_corresp)
        sup_idx_face = np.random.randint(0, face_data.shape[0], self.n_supervision_points)
        sup_idx_non_face = np.random.randint(0, non_face_data.shape[0], self.n_supervision_outer)
        sup_idx_off_surface = np.random.randint(0, off_surface_data.shape[0], self.n_supervision_off)
        neutral_sup_idx_face = np.random.randint(0, neutral_face_data.shape[0], self.n_supervision_points)
        neutral_sup_idx_non_face = np.random.randint(0, neutral_non_face_data.shape[0], self.n_supervision_outer)
        neutral_sup_idx_off_surface = np.random.randint(0, neutral_off_surface_data.shape[0], self.n_supervision_off)
        sup_points_neutral = point_corresp[sup_idx_corresp, :3]
        sup_points_posed = point_corresp[sup_idx_corresp, 3:]

        # sample points in posed scan
        surf_points_samples_face = face_data[sup_idx_face, :3] / 25
        surf_normals_samples_face = face_data[sup_idx_face, 3:6]

        surf_points_samples_non_face = non_face_data[sup_idx_non_face, :3] / 25
        surf_normals_samples_non_face = non_face_data[sup_idx_non_face, 3:6]

        off_surf_points_samples = off_surface_data[sup_idx_off_surface, :3] / 25
        off_surf_normals_samples = off_surface_data[sup_idx_off_surface, 3:6]
        off_surf_sdf_samples = off_surface_data[sup_idx_off_surface, 9:] / 25


        # for corresponding neutral scan
        neutral_surf_points_samples_face = neutral_face_data[neutral_sup_idx_face, :3] / 25
        neutral_surf_normals_samples_face = neutral_face_data[neutral_sup_idx_face, 3:6]

        neutral_surf_points_samples_non_face = neutral_non_face_data[neutral_sup_idx_non_face, :3] / 25
        neutral_surf_normals_samples_non_face = neutral_non_face_data[neutral_sup_idx_non_face, 3:6]

        neutral_off_surf_points_samples = neutral_off_surface_data[neutral_sup_idx_off_surface, :3] / 25
        neutral_off_surf_normals_samples = neutral_off_surface_data[neutral_sup_idx_off_surface, 3:6]
        neutral_off_surf_sdf_samples = neutral_off_surface_data[neutral_sup_idx_off_surface, 9:] / 25


        gt_anchors = self.anchors[iden]



        sup_grad_far = uniform_ball(self.n_supervision_points // 8, rad=0.5)

        return {'corresp_neutral': sup_points_neutral,
                'corresp_posed': sup_points_posed,

                'points_surface': surf_points_samples_face,
                'normals_surface': surf_normals_samples_face,
                'points_surface_outer': surf_points_samples_non_face,
                'normals_surface_outer': surf_normals_samples_non_face,
                'points_off_surface': off_surf_points_samples,
                'normals_off_surface': off_surf_normals_samples,
                'sdfs_off_surface': off_surf_sdf_samples,

                'points_neutral': neutral_surf_points_samples_face,
                'normals_neutral': neutral_surf_normals_samples_face,
                'points_neutral_outer': neutral_surf_points_samples_non_face,
                'normals_neutral_outer': neutral_surf_normals_samples_non_face,
                'points_off_surface_neutral': neutral_off_surf_points_samples,
                'normals_off_surface_neutral': neutral_off_surf_normals_samples,
                'sdfs_off_surface_neutral': neutral_off_surf_sdf_samples,

                'sup_grad_far': sup_grad_far,

                'idx': np.array([idx]),
                'iden': np.array([self.subjects.index(iden)]),
                'expr': np.array([expr]),
                'subj_ind': np.array([subj_ind]),
                'gt_anchors': gt_anchors,
                'is_neutral': np.array([is_neutral])}

    def get_loader(self, shuffle=True):
        #random.seed(1)
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

