import os
import numpy as np
import pyvista as pv

from NPHM.data.manager import DataManager
from NPHM import env_paths
from NPHM.evaluation.render_utils import fibonacci_sphere, m3dLookAt, render_glcam, get_3d_points


def main(RENDER_BACK):
    subjects_test = env_paths.subjects_test

    manager = DataManager()

    for subject in subjects_test:
        save_dir = env_paths.DATA_SINGLE_VIEW
        os.makedirs(save_dir, exist_ok=True)

        expressions = manager.get_expressions(subject, testing=True)
        if RENDER_BACK:
            expressions = expressions[:1]
        for expression in expressions:

            res_factor = 1
            crop = 0
            resolution = 1000 // res_factor
            rend_size = (resolution, resolution)

            K = np.array(
                [[1500. / res_factor, 0.00000000e+00, (rend_size[1] // 2) - crop],
                 [0.00000000e+00, 1500 / res_factor, (rend_size[0] // 2) - crop],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
            )
            rend_size = (rend_size[0] - 2 * crop, rend_size[1] - 2 * crop)

            mesh2 = manager.get_raw_mesh(subject, expression)

            if RENDER_BACK:
                angle = np.pi
                eye = np.array([np.sin(angle), 0, np.cos(angle)]) * 0.65
                eye[1] += 0.4
            else:
                eyes = np.array(fibonacci_sphere(1000))
                eyes = eyes[eyes[:, 2] > 0.6, :]
                eyes = eyes[eyes[:, 2] < 0.9, :]
                eyes = eyes[eyes[:, 1] < 0.55, :]
                eyes = eyes[eyes[:, 1] > -0.55, :]

                ii = np.random.randint(0, len(eyes))
                print(eyes[ii])
                eye = eyes[ii]*0.65
            E = m3dLookAt(eye * 4,
                          np.zeros([3]),
                          np.array([0, 1, 0]))

            depth, normals = render_glcam(mesh2, K, E, rend_size=rend_size,
                                          znear=0.2, zfar=5.0)


            points = get_3d_points(depth, K, E, rend_size=rend_size, znear=0.2, zfar=5.0)


            #pl = pv.Plotter()
            #pl.add_mesh(mesh2)
            #pl.add_points(points)
            #pl.show()


            export_dir = manager.get_single_view_dir(subject, expression)
            os.makedirs(export_dir, exist_ok=True)
            export_path = manager.get_single_view_path(subject, expression, full_depth_map=True, is_back=RENDER_BACK)
            np.save(export_path, points)

            above = manager.cut_throat(points, subject, expression)
            points = points[above, :]
            rnd_idx = np.random.randint(0, points.shape[0], 2500)
            points = points[rnd_idx, :]

            #pl = pv.Plotter()
            #pl.add_mesh(mesh2)
            #pl.add_points(points)
            #pl.show()

            export_path = manager.get_single_view_path(subject, expression, full_depth_map=False, is_back=RENDER_BACK)
            np.save(export_path, points)



if __name__ == '__main__':
    main(RENDER_BACK=False)
    main(RENDER_BACK=True)