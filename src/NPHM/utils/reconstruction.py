import numpy as np
import trimesh
import mcubes

def create_grid_points_from_bounds(minimun, maximum, res, scale=None):
    if scale is not None:
        res = int(scale * res)
        minimun = scale * minimun
        maximum = scale * maximum
    x = np.linspace(minimun[0], maximum[0], res)
    y = np.linspace(minimun[1], maximum[1], res)
    z = np.linspace(minimun[2], maximum[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def mesh_from_logits(logits, mini, maxi, resolution):
    logits = np.reshape(logits, (resolution,) * 3)

    logits *= -1

    # padding to ba able to retrieve object close to bounding box bondary
    # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1000)
    threshold = 0.0
    vertices, triangles = mcubes.marching_cubes(logits, threshold)

    # rescale to original scale
    step = (np.array(maxi) - np.array(mini)) / (resolution - 1)
    vertices = vertices * np.expand_dims(step, axis=0)
    vertices += [mini[0], mini[1], mini[2]]

    return trimesh.Trimesh(vertices, triangles)
