import numpy as np
import pyvista as pv
import pathlib
import trimesh
import pyrender
import math

KK = np.array([
    [2440, 0, 480],
    [0, 2440, 640],
    [0, 0, 1]], dtype=np.float32)

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(str(pathlib.Path(__file__).parent.resolve().absolute()) + "/shaders/mesh.vert",
                                                                 str(pathlib.Path(__file__).parent.resolve().absolute()) + "/shaders/mesh.frag",
                                                                 defines=defines)
        return self.program
    def clear(self):
        self.program = None

def render_glcam(model_in,  # model name or trimesh
                 K,
                 Rt,
                 rend_size=(512, 512),
                 znear=0.1,
                 zfar=2.0):

    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # Scene creation
    scene = pyrender.Scene(ambient_light = [0.45,0.45,0.45, 1.0])

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)

    scene.add(cam, pose=Rt)

    # Set up the light
    instensity = 0.75

    light1 = pyrender.PointLight(intensity=instensity)
    light2 = pyrender.PointLight(intensity=instensity)

    light_pose1 = m3dLookAt(Rt[:3, 3]/2 + np.array([0, 0, 300]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))
    light_pose2 = m3dLookAt(Rt[:3, 3]/2 + np.array([0, 0, 0]), np.mean(mesh.vertices, axis=0), up= np.array([0, 1, 0]))

    light_pose1[:3, 3] = Rt[:3, 3]/2 + np.array([0.15, 0.1, -0.15])
    light_pose2[:3, 3] = Rt[:3, 3]/2 + np.array([-0.15, 0.1, -0.15])


    scene.add(light1, pose=light_pose1)
    scene.add(light2, pose=light_pose2)


    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)

    r._renderer._program_cache = CustomShaderCache()

    normals, depth = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    r.delete()
    world_space_normals = normals / 255 * 2 - 1

    depth[depth == 0] = float('inf')
    depth = (zfar + znear - (2.0 * znear * zfar) / depth) / (zfar - znear)

    return depth, world_space_normals


def get_3d_points(depth, K, Rt, rend_size=(512, 512), normals=None, znear=0.1, zfar=2.0):
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0], K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=znear, zfar=zfar)


    xx, yy = np.meshgrid(np.arange(rend_size[0]), np.arange(rend_size[1]))
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    pixel_inds = np.stack([xx, yy], axis=-1).astype(np.int32)
    lms3d = unproject_points(pixel_inds[:, :2], depth, rend_size, cam.get_projection_matrix(rend_size[1], rend_size[0]), Rt)

    return lms3d


def unproject_points(ppos, depth, rend_size, K, Rt):
    points = np.ones((ppos.shape[0], 4))
    points[:, [1, 0]] = ppos.astype(float)
    points[:, 0] = points[:, 0] / (rend_size[1] - 1) * 2 - 1
    points[:, 1] = points[:, 1] / (rend_size[0] - 1) * 2 - 1

    points[:, 1] *= -1
    ppos[:, 0] = np.clip(ppos[:, 0], 0, rend_size[0])
    ppos[:, 1] = np.clip(ppos[:, 1], 0, rend_size[1])
    points_depth = depth[ppos[:, 0], ppos[:, 1]]
    points[:, 2] = points_depth
    depth_cp = points[:, 2].copy()
    clipping_to_world = np.matmul(Rt, np.linalg.inv(K))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]
    points = points[:, :3]

    points[depth_cp >= 1, :] = np.NaN

    return points



def m3dLookAt(eye, target, up):
    mz = (eye-target)
    mz /= np.linalg.norm(mz, keepdims=True)  # inverse line of sight
    mx = np.cross(up, mz)
    mx /= np.linalg.norm(mx, keepdims=True)
    my = np.cross(mz, mx)
    my /= np.linalg.norm(my)
    tx = eye[0] #np.dot(mx, eye)
    ty = eye[1] #np.dot(my, eye)
    tz = eye[2] #-np.dot(mz, eye)
    return np.array([[mx[0], my[0], mz[0], tx],
                     [mx[1], my[1], mz[1], ty],
                     [mx[2], my[2], mz[2], tz],
                     [0, 0, 0, 1]])


def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def gen_render_samples(m, N, scale=4):
    m = m.copy()
    m.vertices /= scale
    cams = fibonacci_sphere(N + 2)[1:-1]
    cams.reverse()
    all_points = []
    all_normals = []
    for cam_origin in cams:

        E = m3dLookAt(np.array(cam_origin) * 0.6,
                      np.zeros([3]),
                      np.array([0, 1, 0]))

        depth, normals = render_glcam(m, KK, E, rend_size=(1280, 960))
        points3d = get_3d_points(depth, KK, E, rend_size=(1280, 960))

        valid = np.logical_not(np.any(np.isnan(points3d), axis=-1))
        points3d = points3d[valid, :]

        normals = np.transpose(normals, [1, 0, 2])
        normals = normals.reshape([-1, 3])
        normals = normals[valid, :]

        # back face removal
        ray_dir = points3d - np.array(cam_origin) * 0.6
        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
        angle = np.sum(ray_dir * normals, axis=-1)

        all_points.append(points3d[angle < -0.01, :])
        all_normals.append(normals[angle < -0.01, :])

    return np.concatenate(all_points, axis=0)*scale,\
           np.concatenate(all_normals, axis=0)


if __name__ == '__main__':
    import distinctipy

    N = 20
    m = trimesh.load('/mnt/hdd/NRR_FLAME/andrei/expression_2/warped.ply', process=False)

    all_points, all_normals = gen_render_samples(m, N, scale=1)
    colors = distinctipy.get_colors(N)
    pl = pv.Plotter()
    pl.add_mesh(m)
    #for i, p3d in enumerate(all_points):
        # pl.add_points(p3d, color=colors[i])
    pl.add_points(all_points, scalars=all_normals[:, 0])
    #    print(p3d.shape)
    pl.show()

