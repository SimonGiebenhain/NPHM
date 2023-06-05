import trimesh
import numpy as np
import os
from multiprocessing import Pool

from NPHM import env_paths
from NPHM.data.manager import DataManager
from NPHM.utils.mesh_operations import cut_trimesh_vertex_mask





def sample(m_neutral, m, std, n_samps):
    p_neureal, idx_neutral = m_neutral.sample(n_samps, return_index=True)
    normals_neutral = m_neutral.face_normals[idx_neutral, :]
    faces = m_neutral.faces[idx_neutral]
    faces_lin = np.reshape(faces, [-1])
    triangles_neutral = m_neutral.vertices[faces_lin, :]
    triangles_neutral = np.reshape(triangles_neutral, [-1, 3, 3])

    bary = trimesh.triangles.points_to_barycentric(triangles_neutral, p_neureal, method='cross')
    offsets = np.random.randn(p_neureal.shape[0]) * std
    offsets = np.expand_dims(offsets, axis=-1)
    p_neureal += offsets * normals_neutral

    faces = m.faces[idx_neutral]
    normals = m.face_normals[idx_neutral, :]
    faces_lin = np.reshape(faces, [-1])
    triangles = m.vertices[faces_lin, :]
    triangles = np.reshape(triangles, [-1, 3, 3])

    p = trimesh.triangles.barycentric_to_points(triangles, bary)
    p += offsets * normals
    return p_neureal, p, normals_neutral, normals


def main(s):
    manager = DataManager()
    expressions = manager.get_expressions(subject=s)
    for expression in expressions:
        if not VIZ and os.path.exists(manager.get_train_path_deformation(s, expression, rnd_file=env_paths.NUM_SPLITS_EXPR-1)):
            print('skip')
            continue


        n_expr = manager.get_neutral_expression(s, neutral_type='open')
        if n_expr is None:
            continue
        m_neutral = manager.get_registration_mesh(subject=s,
                                                  expression=n_expr,
                                                  )
        m = manager.get_registration_mesh(subject=s, expression=expression)

        #pl = pv.Plotter()
        #pl.add_mesh(face_region_template)
        #pl.add_mesh(m, color='red')
        #pl.show()

        invalid = face_region_template.visual.vertex_colors[:, 0] != 255
        m = cut_trimesh_vertex_mask(m, np.logical_not(invalid))
        m_neutral = cut_trimesh_vertex_mask(m_neutral, np.logical_not(invalid))



        if VIZ:
            pl = pv.Plotter()
            pl.add_mesh(m)
            pl.add_mesh(m_neutral, color='red')
            pl.show()

        p_neutral, p, normals_neutral, normals = sample(m_neutral, m, 0.01, n_samps=N_SAMPLES)#0.01)
        p_neutral2, p2, normals_neutral2, normals2 = sample(m, m_neutral, 0.01, n_samps=N_SAMPLES)#0.01)
        p_neutral = np.concatenate([p_neutral, p2], axis=0)
        p = np.concatenate([p, p_neutral2], axis=0)
        normals_neutral = np.concatenate([normals_neutral, normals2], axis=0)
        normals = np.concatenate([normals, normals_neutral2], axis=0)

        p_neutral_tight, p_tight, normals_neutral_tight, normals_tight = sample(m_neutral, m, 0.002, n_samps=N_SAMPLES)#0.002)
        p_neutral_tight2, p_tight2, normals_neutral_tight2, normals_tight2 = sample(m, m_neutral, 0.002, n_samps=N_SAMPLES)#0.002)
        p_neutral_tight = np.concatenate([p_neutral_tight, p_tight2], axis=0)
        p_tight = np.concatenate([p_tight, p_neutral_tight2], axis=0)
        normals_neutral_tight = np.concatenate([normals_neutral_tight, normals_tight2], axis=0)
        normals_tight = np.concatenate([normals_tight, normals_neutral_tight2], axis=0)


        all_p_neutral = np.concatenate([p_neutral, p_neutral_tight], axis=0)
        all_normals_neutral = np.concatenate([normals_neutral, normals_neutral_tight], axis=0)
        all_p = np.concatenate([p, p_tight], axis=0)
        all_normals = np.concatenate([normals, normals_tight], axis=0)
        perm = np.random.permutation(all_p.shape[0])
        all_p_neutral = all_p_neutral[perm, :]
        all_normals_neutral = all_normals_neutral[perm, :]
        all_p = all_p[perm, :]
        all_normals = all_normals[perm, :]
        if np.any(np.isnan(all_p)) or np.any(np.isnan(all_normals)):
            print('DONE')
            break

        if VIZ:
            pl = pv.Plotter(shape=(1, 2))
            pl.subplot(0, 0)
            pl.add_points(all_p_neutral, scalars=all_normals_neutral[:, 0])
            pl.subplot(0, 1)
            pl.add_points(all_p, scalars=all_normals_neutral[:, 0])
            pl.link_views()
            pl.show()
        data = np.concatenate([all_p_neutral, all_p], axis=-1)
        data_normals = np.concatenate([all_normals_neutral, all_normals], axis=-1)
        split_files = np.array_split(data, env_paths.NUM_SPLITS_EXPR, axis=0)
        #split_files_normals = np.array_split(data_normals, 100, axis=0)
        if not VIZ:
            export_dir_se = manager.get_train_dir_deformation(s, expression)
            os.makedirs(export_dir_se, exist_ok=True)
            for i in range(len(split_files)):
                split_file_path = manager.get_train_path_deformation(s, expression, rnd_file=i)
                np.save(split_file_path, split_files[i])
            #for i in range(len(split_files_normals)):
            #    split_file_path = export_dir + '/{}_{:03d}/corresp_normals_{}.npy'.format(s, expression, i)
            #    np.save(split_file_path, split_files[i])


        if VIZ and False:
            deform = all_p - all_p_neutral


            #deform -= deform.min()
            #deform /= deform.max()
            #deform *= 255
            #deform = deform.astype(np.uint8)
            #color = deform

            deform_norm = np.linalg.norm(deform, axis=-1)
            deform_norm -= np.min(deform_norm)
            deform_norm /= np.max(deform_norm)
            deform_norm *= 255
            color = np.zeros([deform_norm.shape[0], 4], dtype=np.uint8)
            color[:, 0] = deform_norm.astype(np.uint8)
            color[:, 3] = 255

            pc_neutral = trimesh.points.PointCloud(all_p_neutral)
            pc = trimesh.points.PointCloud(all_p, colors=color)

            pl = pv.Plotter()
            for i in range(1000):
                pl.add_mesh(pv.Line(all_p_neutral[i, :], all_p[i, :]))
            #pl.add_points(all_p, scalars=all_normals_neutral[:, 0])  # color)
            #pl.link_views()
            pl.camera_position = 'xy'
            pl.camera.position = (0, 0, 3)
            #pl.camera_set = True
            pl.show()

            pl = pv.Plotter(shape=(1, 2))
            pl.subplot(0, 0)

            pl.add_mesh(m_neutral)
            #pl.add_mesh(pv.Plane((0, 0, 0), (1, 0, 0), i_size=2, j_size=10))

            pl.subplot(0, 1)
            pl.add_mesh(m)
            pl.add_points(all_p, scalars=all_normals_neutral[:, 0])#color)
            #pl.add_mesh(pv.Plane((0, 0, 0), (1, 0, 0), i_size=2, j_size=10))
            pl.link_views()
            pl.show()

VIZ = False

if VIZ:
    import pyvista as pv

N_SAMPLES = 250000

if __name__ == '__main__':
    manager = DataManager()

    all_subjects = manager.get_all_subjects()

    print(f"FOUND {len(all_subjects)} subjects!")

    export_dir = env_paths.SUPERVISION_DEFORMATION_OPEN
    os.makedirs(export_dir, exist_ok=True)

    face_region_template = trimesh.load(env_paths.ASSETS + '/template_face_up.ply', process=False)

    if not VIZ:
        p = Pool(10)
        p.map(main, all_subjects)
        p.close()
        p.join()
    else:
        main(all_subjects[0])


