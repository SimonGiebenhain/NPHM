import os

import point_cloud_utils as pcu

from NPHM import env_paths
import trimesh
from typing import Literal, Union, Dict, List, Optional
import numpy as np



class DataManager():
    def __init__(self, dummy_path = None):

        if dummy_path is not None:
            env_paths.DATA = dummy_path + '/dataset/'
            env_paths.DATA_SINGLE_VIEW = dummy_path + '/single_view/'

        self.lm_inds_upsampled = np.array([2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
                                           3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
                                           3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
                                           3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
                                           1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
                                           1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
                                           3509, 2786])

        self.anchor_indices = np.array([2712, 1579, 3485, 3756, 3430, 3659, 2711, 1575, 338, 27, 3631,
                                        3832, 2437, 1175, 3092, 2057, 3422, 3649, 3162, 2143, 617, 67,
                                        3172, 2160, 2966, 1888, 1470, 2607, 1896, 2981, 3332, 3231, 3494,
                                        3526, 3506, 3543, 3516, 3786, 3404])


    def get_all_subjects(self) -> List[int]:
        all_subjects = [int(pid) for pid in os.listdir(env_paths.DATA) if pid.isdigit()]
        all_subjects.sort()
        return all_subjects


    def get_train_subjects(self,
                           neutral_type: Literal['open', 'closed'] = 'open',
                           exclude_missing_neutral : bool = True) -> List[int]:
        all_subjects = self.get_all_subjects()
        non_train = env_paths.subjects_test + env_paths.subjects_eval
        train_subjects =  [s for s in all_subjects if s not in non_train]
        if exclude_missing_neutral:
            train_subjects = [s for s in train_subjects if self.get_neutral_expression(s, neutral_type) is not None]
        return train_subjects


    def get_eval_subjects(self,
                          neutral_type: Literal['open', 'closed'] = 'open',
                          exclude_missing_neutral: bool = True) -> List[int]:
        eval_subjects = env_paths.subjects_eval
        if exclude_missing_neutral:
            eval_subjects = [s for s in eval_subjects if self.get_neutral_expression(s, neutral_type) is not None]
        return eval_subjects


    def get_test_subjects(self) -> List[int]:
        return env_paths.subjects_test


    def get_expressions(self,
                        subject : int,
                        testing : bool = False,
                        exclude_bad_scans : bool = True) -> List[int]:
        expressions = [int(f) for f in os.listdir(self.get_subject_dir(subject))]
        expressions.sort()
        if testing:
            expressions = [ex for ex in expressions if not (subject in env_paths.invalid_expressions_test and
                                                            ex in env_paths.invalid_expressions_test[subject])]
        if exclude_bad_scans:
            expressions = [ex for ex in expressions if not(subject in env_paths.bad_scans and ex in env_paths.bad_scans[subject])]
        return expressions


    def get_neutral_expression(self,
                               subject : int,
                               neutral_type : Literal['open', 'closed'] = 'open'
                               ) -> Optional[int]:
        if neutral_type == 'open':
            if subject not in env_paths.neutrals:
                return None
            neutral_expression = env_paths.neutrals[subject]
            if neutral_expression >= 0:
                return neutral_expression
            else:
                return None
        elif neutral_type == 'closed':
            if subject not in env_paths.neutrals:
                return None
            neutral_expression = env_paths.neutrals_closed[subject]
            if neutral_expression >= 0:
                return neutral_expression
            else:
                return None
        else:
            raise TypeError(f'Unknown neutral type {neutral_type} encountered! Expected on of [open, closed]!')


    def get_scan_dir(self,
                      subject : int,
                      expression : int) -> str:
        return f"{env_paths.DATA}/{subject:03d}/{expression:03d}/"

    def get_subject_dir(self,
                      subject : int) -> str:
        return f"{env_paths.DATA}/{subject:03d}/"


    def get_raw_path(self,
                     subject: int,
                     expression: int) -> str:
        return f"{self.get_scan_dir(subject, expression)}/scan.ply"


    def get_flame_path(self,
                     subject: int,
                     expression: int) -> str:
        return f"{self.get_scan_dir(subject, expression)}/flame.ply"

    def get_registration_path(self,
                     subject: int,
                     expression: int) -> str:
        return f"{self.get_scan_dir(subject, expression)}/registration.ply"

    def get_transform_from_metric(self,
                                  subject : int,
                                  expression : int) -> Dict[str, np.ndarray]:
        data_dir = self.get_scan_dir(subject, expression)
        s = np.load(f"{data_dir}/s.npy")
        R = np.load(f"{data_dir}/R.npy")
        t = np.load(f"{data_dir}/t.npy")
        return {"s": s, "R": R, "t": t}

    def get_raw_mesh(self,
                     subject : int,
                     expression : int,
                     coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                     mesh_type : Literal['trimesh', 'pcu'] = 'trimesh',
                     textured : bool = False # only relevant for mesh_type='pcu'
                     ) -> Union[trimesh.Trimesh, pcu.TriangleMesh]:

        raw_path = self.get_raw_path(subject, expression)
        if mesh_type == 'trimesh':
            m_raw = trimesh.load(raw_path, process=False)
        else:
            if textured:
                m_raw = pcu.load_triangle_mesh(raw_path)
            else:
                m_raw = pcu.TriangleMesh()
                v, f = pcu.load_mesh_vf(raw_path)
                m_raw.vertex_data.positions = v
                m_raw.face_data.vertex_ids = f

        if coordinate_system == 'flame':
            m_raw = self.transform_nphm_2_flame(m_raw)
        if coordinate_system == 'raw':
            m_raw = self.transform_nphm_2_raw(m_raw, subject, expression)

        return m_raw


    def get_flame_mesh(self,
                       subject : int,
                       expression : int,
                       coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                       mesh_type : Literal['trimesh', 'pcu'] = 'trimesh'
                       )-> Union[trimesh.Trimesh, pcu.TriangleMesh]:

        flame_path = self.get_flame_path(subject, expression)
        if mesh_type == 'trimesh':
            m_flame = trimesh.load(flame_path, process=False)
        else:
            m_flame = pcu.TriangleMesh()
            v, f = pcu.load_mesh_vf(flame_path)
            m_flame.vertex_data.positions = v
            m_flame.face_data.vertex_ids = f

        if coordinate_system == 'flame':
            m_flame = self.transform_nphm_2_flame(m_flame)
        if coordinate_system == 'raw':
            m_flame = self.transform_nphm_2_raw(m_flame, subject, expression)

        return m_flame

    def get_registration_mesh(self,
                              subject : int,
                              expression : int,
                              coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                              mesh_type: Literal['trimesh', 'pcu'] = 'trimesh'
                              ) -> Union[trimesh.Trimesh, pcu.TriangleMesh]:

        regi_path = self.get_registration_path(subject, expression)
        if mesh_type == 'trimesh':
            mesh = trimesh.load(regi_path, process=False)
        else:
            mesh = pcu.TriangleMesh()
            v, f = pcu.load_mesh_vf(regi_path)
            mesh.vertex_data.positions = v
            mesh.face_data.vertex_ids = f

        if coordinate_system == 'flame':
            mesh = self.transform_nphm_2_flame(mesh)
        if coordinate_system == 'raw':
            mesh = self.transform_nphm_2_raw(mesh, subject, expression)

        return mesh


    def get_landmarks(self,
                      subject : int,
                      expression : int,
                      coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm') -> np.ndarray:
        fine_mesh = self.get_registration_mesh(subject, expression, coordinate_system)
        landmarks = fine_mesh.vertices[self.lm_inds_upsampled, :]
        return landmarks

    def get_facial_anchors(self,
                      subject : int,
                      expression : int,
                      coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm') -> np.ndarray:
        fine_mesh = self.get_registration_mesh(subject, expression, coordinate_system)
        anchors = fine_mesh.vertices[self.anchor_indices, :]
        return np.array(anchors)



    def get_single_view_obs(self,
                            subject : int,
                            expression : int,
                            include_back : bool = True,
                            coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                            disable_cut_throat = False,
                            full_obs = False
                            ) -> np.ndarray:

        points = np.load(self.get_single_view_path(subject, expression, full_depth_map=full_obs))
        if include_back:
            back_path = self.get_single_view_path(subject, expression, full_depth_map=full_obs, is_back=True)
            if not os.path.exists(back_path):
                print('WARNING: observation from back not available!')
            else:
                points_back = np.load(back_path)
                points = np.concatenate([points, points_back], axis=0)

        if not disable_cut_throat:
            above = self.cut_throat(points, subject, expression)
            points = points[above, :]

        if coordinate_system == 'flame':
            points = self.transform_nphm_2_flame(points)
        if coordinate_system == 'raw':
            points = self.transform_nphm_2_raw(points, subject, expression)

        return points


    def cut_throat(self,
                   points : np.ndarray,
                   subject : int,
                   expression : int,
                   coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                   margin : float = 0) -> np.ndarray:

            template_mesh = self.get_flame_mesh(subject, expression, coordinate_system=coordinate_system)
            idv1 = 3276
            idv2 = 3207
            idv3 = 3310
            v1 = template_mesh.vertices[idv1, :]
            v2 = template_mesh.vertices[idv2, :]
            v3 = template_mesh.vertices[idv3, :]
            origin = v1
            line1 = v2 - v1
            line2 = v3 - v1
            normal = np.cross(line1, line2)

            direc = points - origin
            angle = np.sum(normal * direc, axis=-1)
            above = angle > margin
            return above


    ####################################################################################################################
    #### Transformations between cooridnate systems ####
    ####################################################################################################################

    def transform_nphm_2_flame(self,
                               object : Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]
                               ) -> Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]:

        if isinstance(object, np.ndarray):
            object /= 4
        elif isinstance(object, trimesh.Trimesh):
            object.vertices /= 4
        elif isinstance(object, pcu.TriangleMesh):
            object.vertex_data.positions /= 4
        else:
            raise TypeError(f'Unexpected type encountered in coordinate transormation. \n '
                            'Expected one of [trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]. '
                            f'But found {type(object)}')

        return object


    def transform_nphm_2_raw(self,
                             object : Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh],
                             subject : int,
                             expression : int
                             ) -> Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]:

        transform = self.get_transform_from_metric(subject, expression)

        if isinstance(object, np.ndarray):
            object = 1/transform['s'] * (object - transform['t']) @ transform['R']
        elif isinstance(object, trimesh.Trimesh):
            object.vertices = 1/transform['s'] * (object.vertices - transform['t']) @ transform['R']
        elif isinstance(object, pcu.TriangleMesh):
            object.vertex_data.positions = 1/transform['s'] * (object.vertex_data.positions - transform['t']) @ transform['R']
        else:
            raise TypeError(f'Unexpected type encountered in coordinate transormation. \n '
                            'Expected one of [trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]. '
                            f'But found {type(object)}')

        return object


    ####################################################################################################################
    ######### get paths relevant for training ###########
    ####################################################################################################################


    def get_train_dir_identity(self,
                               subject : int) -> str:
        return f"{env_paths.SUPERVISION_IDENTITY}/{subject:03d}/"


    def get_train_path_identity_face(self,
                                subject: int,
                                expression: int,
                                rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = np.random.randint(0, env_paths.NUM_SPLITS)
        return f"{self.get_train_dir_identity(subject)}/{expression}_{rnd_file}_face.npy"


    def get_train_path_identity_non_face(self,
                                subject: int,
                                expression: int,
                                rnd_file: Optional[int] = None
                                ) -> str:
        if rnd_file is None:
            rnd_file = np.random.randint(0, env_paths.NUM_SPLITS)
        return f"{self.get_train_dir_identity(subject)}/{expression}_{rnd_file}_non_face.npy"


    def get_train_dir_deformation(self,
                                  subject : int,
                                  expression : int) -> str:
        return f"{env_paths.SUPERVISION_DEFORMATION_OPEN}/{subject:03d}/{expression:03d}/"


    def get_train_path_deformation(self,
                                   subject : int,
                                   expression : int,
                                   rnd_file : Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = np.random.randint(0, env_paths.NUM_SPLITS_EXPR)
        return f"{self.get_train_dir_deformation(subject, expression)}/corresp_{rnd_file}.npy"


    def get_single_view_dir(self,
                            subject : int,
                            expression : int):
        return f"{env_paths.DATA_SINGLE_VIEW}/{subject:03d}/{expression}"


    def get_single_view_path(self,
                             subject : int,
                             expression : int,
                             full_depth_map = False,
                             is_back = False):
        dir_name = self.get_single_view_dir(subject, expression)
        if not full_depth_map:
            if is_back:
                return f"{dir_name}/obs_back.npy"
            else:
                return f"{dir_name}/obs.npy"
        else:
            if is_back:
                return f"{dir_name}/full_obs_back.npy"
            else:
                return f"{dir_name}/full_obs.npy"




