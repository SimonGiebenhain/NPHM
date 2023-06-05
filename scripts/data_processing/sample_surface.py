import os

import point_cloud_utils as pcu
import numpy as np
import pyvista as pv
import trimesh
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image
#import env_paths
import os.path as osp
import traceback
import tyro
from multiprocessing import Pool

from NPHM.data.manager import DataManager
from NPHM import env_paths



def get_color(tex_path, uv_coords):
    img = np.array(Image.open(tex_path))
    imw = img.shape[1]
    imh = img.shape[0]
    u = np.clip((uv_coords[:, 0] * (imw-1)).astype(int), 0, imw-1)
    v = np.clip((uv_coords[:, 1] * (imh-1)).astype(int), 0, imh-1)

    colors = img[imh-1- v, u]
    return colors


def sample_fields(n_samps, n_samps_off, sigma, s, e):
    manager = DataManager()
    mesh = manager.get_raw_mesh(s, e, mesh_type='pcu', textured=True)


    face_region_mesh = manager.get_registration_mesh(s, e, mesh_type='pcu')
    #face_region_mesh = manager.get_registration_mesh(s, e, mesh_type='trimesh')
    #face_region_mesh.visual.vertex_colors = np.zeros([face_region_mesh.vertices.shape[0], 3], dtype=int)
    #face_region_mesh.visual.vertex_colors[mask, 0] = 255
    #face_region_mesh.show()
    face_region_mesh_vc = face_region_template.vc.copy()

    if VISUALIZE:
        m_raw = trimesh.Trimesh(mesh.vertex_data.positions,
                                mesh.face_data.vertex_ids, process=False)
        m_regi = trimesh.Trimesh(face_region_mesh.vertex_data.positions,
                                face_region_mesh.face_data.vertex_ids, process=False)

        pl = pv.Plotter()
        pl.add_mesh(m_raw)
        pl.add_mesh(m_regi, color='red')
        pl.show()

    texture_path = osp.join(osp.dirname(manager.get_raw_path(s, e)), mesh.textures[0])

    n = pcu.estimate_mesh_vertex_normals(mesh.vertex_data.positions, mesh.face_data.vertex_ids)

    # Generate random samples on the mesh (v, f, n)cut
    # f_i are the face indices of each sample and bc are barycentric coordinates of the sample within a face
    f_i, bc = pcu.sample_mesh_random(mesh.vertex_data.positions, mesh.face_data.vertex_ids, num_samples=n_samps)

    # compute field values for points on the surface
    surf_points = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, f_i, bc, mesh.vertex_data.positions)
    surf_normals = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, f_i, bc, n)
    #surf_uv_coords = (mesh.face_data.wedge_texcoords[f_i] * bc[:, :, np.newaxis]).sum(1)
    #surf_colors = get_color(texture_path, surf_uv_coords)

    above = manager.cut_throat(surf_points, s, e)

    surf_points = surf_points[above, :]
    surf_normals = surf_normals[above, :]
    #surf_colors = surf_colors[above, :]

    # determine which points lie in the facial region
    if face_region_mesh is not None:
        face_region_mesh.vertex_data.positions = face_region_mesh.vertex_data.positions[mask, :]
        non_face_vertices = np.arange(mask.shape[0])[~mask]
        good_faces = ~np.any(np.isin(face_region_mesh.face_data.vertex_ids, non_face_vertices), axis=-1)

        face_region_mesh.face_data.vertex_ids = face_region_mesh.face_data.vertex_ids[good_faces, :]
        (_, corrs_x_to_y, corrs_y_to_x) = pcu.chamfer_distance(surf_points.astype(np.float32), face_region_mesh.vertex_data.positions, return_index=True, p_norm=2, max_points_per_leaf=10)
        #d_region, fi_region, bc_region = pcu.closest_points_on_mesh(surf_points.astype(np.float32),
        #                                                            face_region_mesh.vertex_data.positions,
        #                                                            face_region_mesh.face_data.vertex_ids)
        #
        #closest_vertex_color_region = pcu.interpolate_barycentric_coords(face_region_mesh.face_data.vertex_ids, fi_region, bc_region,
        #                                                    face_region_mesh_vc)
        #face_region = closest_vertex_color_region[:, 0] == 255
        d_region = np.linalg.norm(surf_points.astype(np.float32) - face_region_mesh.vertex_data.positions[corrs_x_to_y, :], axis=-1)
        face_region = d_region < 5/25

        if VISUALIZE:
            pl = pv.Plotter()
            pl.add_mesh(surf_points[face_region, :], scalars=surf_normals[face_region, 0])
            #pl.add_mesh(trimesh.Trimesh(face_region_mesh.vertex_data.positions,
            #                                                        face_region_mesh.face_data.vertex_ids, process=False))
            pl.add_points(face_region_mesh.vertex_data.positions)
            pl.show()


    #rnd_idx = np.random.randint(0, surf_points.shape[0], n_samps_off)
    #points = surf_points[rnd_idx, :] + np.random.randn(n_samps_off, 3) * sigma


    #sdfs, fi, bc = pcu.signed_distance_to_mesh(points, mesh.vertex_data.positions, mesh.face_data.vertex_ids)

    ### Convert barycentric coordinates to 3D positions
    #normals = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, fi, bc, n)
    #closest_uv = (mesh.face_data.wedge_texcoords[fi] * bc[:, :, np.newaxis]).sum(1)

    #colors = get_color(texture_path, closest_uv)

    if VISUALIZE:
        pl = pv.Plotter()
        pl.add_mesh(trimesh.Trimesh(mesh.vertex_data.positions, mesh.face_data.vertex_ids))
        pl.add_points(surf_points, scalars=surf_normals[:, 0])
        #pl.add_points(surf_points, scalars=surf_colors, rgb=True)
        pl.show()

    if face_region_mesh is not None:
        rnd_idx_non_face = np.random.randint(0, np.sum(~face_region), n_samps_off)
        return {'face': {'points':surf_points[face_region, :],
                         #'colors': surf_colors[face_region, :],
                         'normals': surf_normals[face_region, :]},
                'non-face': {'points': surf_points[~face_region, :][rnd_idx_non_face, :],
                             #'colors': surf_colors[~face_region, :][rnd_idx_non_face, :],
                             'normals': surf_normals[~face_region, :][rnd_idx_non_face, :]},
                #'off-surface': {'points': points,
                #                 'colors': colors,
                #                 'normals': normals,
                #                 'sdfs': sdfs}
                }
    else:
        return surf_points, surf_normals #surf_colors, #points, sdfs,  colors, normals


def run_subject(s):
    manager = DataManager()
    expressions = [manager.get_neutral_expression(subject=s, neutral_type='open')]
    for e in expressions:
        if e is None:
            continue

        if osp.exists(manager.get_train_path_identity_face(s, e, rnd_file=NUM_SPLITS-1)) and not VISUALIZE:
            print('SKIPPING:', s, e)
            return
        try:
            print(s, e)
            N_SAMPS = 25000000
            N_SAMPS_OFF = 1000000
            if VISUALIZE:
                N_SAMPS = N_SAMPS // 10
                N_SAMPS_OFF = N_SAMPS_OFF // 10
            results = sample_fields(N_SAMPS, N_SAMPS_OFF, sigma=1, s=s, e=e)
            if VISUALIZE:
                print(results['face']['points'].shape)
                print(results['non-face']['points'].shape)
                #print(results['off-surface']['points'].shape)

                pl = pv.Plotter()
                pl.add_points(results['face']['points'])
                pl.add_points(results['non-face']['points'], color='yellow')
                #pl.add_points(results['off-surface']['points'], color='green')
                pl.show()

            #data_off = np.concatenate([results['off-surface']['points'],
            #                           results['off-surface']['normals'],
            #                          results['off-surface']['colors'].astype(np.float32),
            #                           results['off-surface']['sdfs'][:, np.newaxis]], axis=1)
            #data_off = data_off.astype(np.float32)
            data_face = np.concatenate([results['face']['points'],
                                        results['face']['normals'],
                                        #results['face']['colors'].astype(np.float32)
                                        ], axis=1)
            data_face = data_face.astype(np.float32)

            data_non_face = np.concatenate([results['non-face']['points'],
                                            results['non-face']['normals'],
                                            #results['non-face']['colors'].astype(np.float32)
                                            ], axis=1)
            data_non_face = data_non_face.astype(np.float32)

            out_dir_s = manager.get_train_dir_identity(s)
            os.makedirs(out_dir_s, exist_ok=True)
            print(out_dir_s, e)
            chunks_face = np.array_split(data_face, NUM_SPLITS, axis=0 )
            chunks_non_face = np.array_split(data_non_face, NUM_SPLITS, axis=0 )
            for i, chunk_face in enumerate(chunks_face):
                np.save(manager.get_train_path_identity_face(s, e, rnd_file=i), chunk_face)
            for i, chunk_non_face in enumerate(chunks_non_face):
                np.save(manager.get_train_path_identity_non_face(s, e, rnd_file=i), chunk_non_face)
            #np.save(osp.join(out_dir_s, '{}_off.npy'.format(e)), data_off)
        except Exception as ex:
            print('EXCEPTION', s, e)
            print(traceback.format_exc())


def main():
    manager = DataManager()

    all_subjects = manager.get_all_subjects()

    print(f"FOUND {len(all_subjects)} subjects!")

    out_dir = env_paths.SUPERVISION_IDENTITY
    os.makedirs(out_dir, exist_ok=True)


    if not VISUALIZE:
        p = Pool(10)
        p.map(run_subject, all_subjects)
        p.close()
        p.join()
    else:
        run_subject(all_subjects[1])


VISUALIZE = False
face_region_template = pcu.load_triangle_mesh(env_paths.ASSETS + '/template_face_up.ply')
mask = np.load(env_paths.ASSETS + 'face.npy')

NUM_SPLITS = env_paths.NUM_SPLITS

if __name__ == '__main__':
    tyro.cli(main)







