import json

import numpy as np
import torch
import trimesh
import os
import os.path as osp
import tyro
from scipy.spatial import cKDTree as KDTree

from NPHM.evaluation.metrics import eval_pointcloud
from NPHM import env_paths
from NPHM.evaluation.render_utils import gen_render_samples
from NPHM.data.manager import DataManager



SHOW_VIZ = False
RESAMPLE = False
if SHOW_VIZ:
    import pyvista as pv
else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


def sq_dist(v1, v2):
    return np.linalg.norm(v1[:, np.newaxis, :] - v2[np.newaxis, :], axis=-1)


def slice_properly(regi, surf_points, extra=None):
    idv1 = 3276
    idv2 = 3207
    idv3 = 3310
    v1 = regi.vertices.copy()[idv1, :]
    v2 = regi.vertices.copy()[idv2, :]
    v3 = regi.vertices.copy()[idv3, :]

    origin = v1
    line1 = v2 - v1
    line2 = v3 - v1
    normal = np.cross(line1, line2)

    #pl = pv.Plotter()
    #pl.add_mesh(regi)
    #pl.add_points(v1, color='yellow', point_size=10)
    #pl.add_points(v2, color='red', point_size=10)
    #pl.add_points(v3, color='red', point_size=10)
    #pl.show()

    direc = surf_points - origin
    angle = np.sum(normal * direc, axis=-1)
    above = angle > 0.003 # add a bit of margin to exclude bottom of reconstructed mesh
    if extra is not None:
        extra = extra[above]
    return surf_points[above], extra



def sample_surface_points(mesh : trimesh.Trimesh,
                          mesh_flame : trimesh.Trimesh,
                          face_idx : np.ndarray,
                          num_samps : int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    samps, samps_normals = gen_render_samples(mesh, 10)
    print('samps', samps.shape)
    samps, samps_normals = slice_properly(mesh_flame, samps, extra=samps_normals)
    print('samps after slice', samps.shape)



    threshold = 0.02 #0.04
    threshold_point2point = 0.04

    kdtree = KDTree(np.array(mesh_flame.vertices[face_idx, :]))
    dist, nn_idx = kdtree.query(samps)
    nn_vertices = mesh_flame.vertices[face_idx, :][nn_idx, :]
    nn_normals = mesh_flame.vertex_normals[face_idx, :][nn_idx, :]
    point2plane_distance = np.abs(np.sum(((samps - nn_vertices) * nn_normals), axis=-1))
    valids = (point2plane_distance <= threshold) & (dist <= threshold_point2point)
    #valids = dist <= threshold



    samps_face = samps[valids, :]
    samps_normals_face = samps_normals[valids, :]

    print('samps_face', samps_face.shape)

    random_idx = np.random.randint(0, samps.shape[0], num_samps)
    random_idx_face = np.random.randint(0, samps_face.shape[0], num_samps)
    points = samps[random_idx, :]
    normals = samps_normals[random_idx, :]
    points_face = samps_face[random_idx_face, :]
    normals_face = samps_normals_face[random_idx_face, :]

    return points, normals, points_face, normals_face

def main(result_dir : str):
    manager = DataManager()
    m_base = trimesh.load(env_paths.ASSETS + 'template.ply', process=False)
    better_face_region = trimesh.load(env_paths.ASSETS + 'better_face_region.ply', process=False)

    sq_dists = np.linalg.norm(m_base.vertices[:, np.newaxis, :] - better_face_region.vertices[np.newaxis, :, :], axis=-1)

    face_idx = np.where(np.any(sq_dists == 0, axis=1))[0]


    EVAL_DIR = result_dir + '/evaluation/'
    os.makedirs(EVAL_DIR, exist_ok=True)

    num_samps = 250000 #1000000

    for subject in env_paths.subjects_test:
        expression_indices = manager.get_expressions(subject, testing=True)
        files = []
        for expression in expression_indices:

            try:
                fine_path = result_dir + '/{}_{}_fine.ply'.format(subject, expression)
                if os.path.exists(fine_path):
                    files.append(fine_path)
                else:
                    files.append(result_dir + '/{}_{}.ply'.format(subject, expression))
            except Exception as e:
                continue

        pairs = list(zip(expression_indices, files))
        pairs.sort(key= lambda p : p[0])
        print('############ EVALUATING ############')
        print(pairs)
        print('####################################')


        if SHOW_VIZ:
            pl = pv.Plotter(shape=(3, 6))
            ei = 0
            for expression, file in pairs:
                if ei >= 6:
                    break

                m_gt = manager.get_raw_mesh(subject,
                                            expression,
                                            coordinate_system='nphm',
                                            )
                m_FLAME = manager.get_flame_mesh(subject,
                                                 expression,
                                                 coordinate_system='nphm',
                                                 )

                m_pred = trimesh.load(file, process=False)
                #m_pred = manager.transform_nphm_2_raw(m_pred, subject, expression)

                input = manager.get_single_view_obs(subject, expression,
                                                    include_back=expression==expression_indices[0],
                                                    coordinate_system='nphm')
                #normals_input = np.load('/mnt/hdd/eval_benchmark/input/normals_{}_{}.npy'.format(subject, expression))
                rnd_idx = np.random.randint(0, input.shape[0], 5000)
                input = input[rnd_idx, :]
                valid_input = manager.cut_throat(input, subject, expression, coordinate_system='nphm')
                input = input[valid_input, :]

                pl.subplot(0, ei)
                pl.add_points(input)

                pl.subplot(1, ei)
                pl.add_mesh(m_pred)

                pl.subplot(2, ei)
                pl.add_mesh(m_gt)

                # pl.add_points(points_GT, color='green')
                # pl.add_points(all_points['NPM_dir'])
                # pl.add_points(all_points['joint'], color='yellow')#, scalars=normals_NPHM[:, 0])
                ei += 1

            pl.link_views()
            pl.camera_position = (0, 0, 6)
            pl.camera.zoom(4)
            pl.camera.roll = 0
            pl.camera.up = (0, 1, 0)
            pl.camera.focal_point = (0, 0, 0)
            pl.show()



        for expression, file in pairs:

            out_dir_gt = f"{env_paths.FITTING_DIR}/GT/{subject}/expression_{expression}/"
            os.makedirs(out_dir_gt, exist_ok=True)

            out_dir = f"{EVAL_DIR}/{subject}/expression_{expression}/"
            os.makedirs(out_dir, exist_ok=True)

            if not SHOW_VIZ and os.path.exists(out_dir + 'metrics.json'):
                print('SKIPPING', subject, expression)
                continue

            print('PROCSSING', subject, expression)

            m_gt = manager.get_raw_mesh(subject, expression, coordinate_system='nphm')
            m_FLAME = manager.get_flame_mesh(subject, expression, coordinate_system='nphm')

            m_pred = trimesh.load(file, process=False)
            #m_pred = manager.transform_nphm_2_raw(m_pred, subject, expression)


            if SHOW_VIZ:
                input = manager.get_single_view_obs(subject, expression,
                                                    include_back=expression==expression_indices[0],
                                                    coordinate_system='nphm')

                pl = pv.Plotter()
                pl.add_mesh(m_pred)
                pl.add_points(input)
                pl.show()




            if os.path.exists(out_dir_gt + 'points.npy') and not RESAMPLE:

                points_GT = np.load(out_dir_gt + 'points.npy')
                normals_GT = np.load(out_dir_gt + 'normals.npy')

                points_GT_face = np.load(out_dir_gt + 'points_face.npy')
                normals_GT_face = np.load(out_dir_gt + 'normals_face.npy')
            else:
                points_GT, normals_GT, points_GT_face, normals_GT_face = sample_surface_points(m_gt,
                                                                                               m_FLAME,
                                                                                               face_idx,
                                                                                               num_samps=num_samps)
                np.save(out_dir_gt + 'points.npy', points_GT)
                np.save(out_dir_gt + 'normals.npy', normals_GT)
                np.save(out_dir_gt + 'points_face.npy', points_GT_face)
                np.save(out_dir_gt + 'normals_face.npy', normals_GT_face)



            if os.path.exists(out_dir + 'points.npy') and not RESAMPLE:

                points = np.load(out_dir + 'points.npy')
                normals = np.load(out_dir + 'normals.npy')

                points_face = np.load(out_dir + 'points_face.npy')
                normals_face = np.load(out_dir + 'normals_face.npy')


            else:

                points, normals, points_face, normals_face = sample_surface_points(m_pred,
                                                                                   m_FLAME,
                                                                                   face_idx,
                                                                                   num_samps=num_samps)
                np.save(out_dir + 'points.npy', points)
                np.save(out_dir + 'normals.npy', normals)
                np.save(out_dir + 'points_face.npy', points_face)
                np.save(out_dir + 'normals_face.npy', normals_face)


            if SHOW_VIZ:
                pl = pv.Plotter(shape=(1, 3))
                input = manager.get_single_view_obs(subject, expression,
                                                    include_back=expression_indices[0],
                                                    coordinate_system='nphm')

                pl.subplot(0, 0)
                pl.add_points(input)#, scalars = normals_input[:, 0])
                pl.add_mesh(m_pred)

                pl.subplot(0, 1)
                pl.add_mesh(m_pred)

                pl.subplot(0, 2)
                pl.add_mesh(m_gt)

                pl.link_views()
                pl.camera_position = (0, 0, 6)
                pl.camera.zoom(4)
                pl.camera.roll = 0
                pl.camera.up = (0, 1, 0)
                pl.camera.focal_point = (0, 0, 0)
                #pl.add_points(points_GT, color='green')
                #pl.add_points(all_points['NPM_dir'])
                #pl.add_points(all_points['joint'], color='yellow')#, scalars=normals_NPHM[:, 0])
                pl.show()


            if SHOW_VIZ:
                points = points[::10, :]
                points_GT = points_GT[::10, :]
                normals = normals[::10, :]
                normals_GT = normals_GT[::10, :]
            metric_tmp, per_point_errors = eval_pointcloud(points,
                                                           points_GT,
                                                           normals,
                                                           normals_GT,
                                                           return_error_pcs=True,
                                                           metric_space=True,
                                                           subject=subject,
                                                           expression=expression)

            if not SHOW_VIZ:
                # Serializing json
                json_object = json.dumps(metric_tmp, indent=4)

                # Writing to sample.json
                with open(out_dir + 'metrics.json', "w") as outfile:
                    outfile.write(json_object)
            else:
                json_object = json.dumps(metric_tmp, indent=4)

                print('head', json_object)


            if SHOW_VIZ:
                points_face = points_face[::10, :]
                points_GT_face = points_GT_face[::10, :]
                normals_face = normals_face[::10, :]
                normals_GT_face = normals_GT_face[::10, :]
            metric_tmp_face, per_point_errors_face = eval_pointcloud(points_face,
                                                                     points_GT_face,
                                                                     normals_face,
                                                                     normals_GT_face,
                                                                     return_error_pcs=True,
                                                                     metric_space=True,
                                                                     subject=subject,
                                                                     expression=expression)

            if not SHOW_VIZ:
                # Serializing json
                json_object = json.dumps(metric_tmp_face, indent=4)

                # Writing to sample.json
                with open(out_dir + 'metrics_face.json', "w") as outfile:
                    outfile.write(json_object)
            else:
                json_object = json.dumps(metric_tmp_face, indent=4)

                print('face', json_object)

            if SHOW_VIZ:
                pl = pv.Plotter(shape=(2, 2))

                pl.subplot(0, 0)
                pl.add_points(points, scalars=per_point_errors['accuracy'])

                pl.subplot(0, 1)
                pl.add_points(points_GT, scalars=per_point_errors['completeness'])

                pl.subplot(1, 0)
                pl.add_points(points_face, scalars=per_point_errors_face['accuracy'])

                pl.subplot(1, 1)
                pl.add_points(points_GT_face, scalars=per_point_errors_face['completeness'])


                pl.link_views()
                pl.camera_position = (0, 0.15, 6)
                pl.camera.zoom(4)
                pl.camera.roll = 0
                pl.camera.up = (0, 1, 0)
                pl.camera.focal_point = (0, 0.15, 0)
                # pl.add_points(points_GT, color='green')
                # pl.add_points(all_points['NPM_dir'])
                # pl.add_points(all_points['joint'], color='yellow')#, scalars=normals_NPHM[:, 0])
                pl.show()

                pl = pv.Plotter(shape=(2, 2))
                pl.subplot(0, 0)
                pl.add_points(points, scalars=per_point_errors['accuracy_normals'])

                pl.subplot(0, 1)
                pl.add_points(points_GT, scalars=per_point_errors['completeness_normals'])

                pl.subplot(1, 0)
                pl.add_points(points_face, scalars=per_point_errors_face['accuracy_normals'])

                pl.subplot(1, 1)
                pl.add_points(points_GT_face, scalars=per_point_errors_face['completeness_normals'])

                pl.link_views()
                pl.camera_position = (0, 0.15, 6)
                pl.camera.zoom(4)
                pl.camera.roll = 0
                pl.camera.up = (0, 1, 0)
                pl.camera.focal_point = (0, 0.15, 0)
                # pl.add_points(points_GT, color='green')
                # pl.add_points(all_points['NPM_dir'])
                # pl.add_points(all_points['joint'], color='yellow')#, scalars=normals_NPHM[:, 0])
                pl.show()



if __name__ == '__main__':
    tyro.cli(main)