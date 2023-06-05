import numpy as np

def cut_trimesh_vertex_mask(mesh, mask):
    invalid_mask = np.logical_not(mask)
    invalid_faces = mesh.vertex_faces[invalid_mask]
    invalid_faces = np.unique(invalid_faces.reshape([-1]))
    invalid_faces = invalid_faces[invalid_faces >= 0]
    invalid_faces_mask = np.zeros(dtype=bool, shape=[mesh.faces.shape[0]])
    invalid_faces_mask[invalid_faces] = 1
    mesh.update_faces(np.logical_not(invalid_faces_mask))
    mesh.remove_unreferenced_vertices()
    return mesh