import numpy as np
from scipy.spatial import cKDTree as KDTree
from NPHM.data.manager import DataManager


# This function is mostly apdopted from occupancy_networks/im2mesh/common.py and occupancy_networks/im2mesh/eval.py
def eval_meshOnet(mesh_pred, gt_mesh, n_points=100000, remove_wall=False, include_cdits_for_viz=False):

    pointcloud_gt, gt_idx = gt_mesh.sample(2 * n_points, return_index=True)
    normals_gt = gt_mesh.face_normals[gt_idx]

    if remove_wall: #! Remove walls and floors
        pointcloud_pred, idx = mesh_pred.sample(2*n_points, return_index=True)
        eps = 0.007
        x_max, x_min = pointcloud_gt[:, 0].max(), pointcloud_gt[:, 0].min()
        y_max, y_min = pointcloud_gt[:, 1].max(), pointcloud_gt[:, 1].min()
        z_max, z_min = pointcloud_gt[:, 2].max(), pointcloud_gt[:, 2].min()

        # add small offsets
        x_max, x_min = x_max + eps, x_min - eps
        y_max, y_min = y_max + eps, y_min - eps
        z_max, z_min = z_max + eps, z_min - eps

        mask_x = (pointcloud_pred[:, 0] <= x_max) & (pointcloud_pred[:, 0] >= x_min)
        mask_y =  (pointcloud_pred[:, 1] >= y_min) # floor
        mask_z = (pointcloud_pred[:, 2] <= z_max) & (pointcloud_pred[:, 2] >= z_min)

        mask = mask_x & mask_y & mask_z
        pointcloud_new = pointcloud_pred[mask]
        # Subsample
        idx_new = np.random.randint(pointcloud_new.shape[0], size=n_points)
        pointcloud_pred = pointcloud_new[idx_new]
        idx = idx[mask][idx_new]
    else:
        pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)
    if include_cdits_for_viz:
        out_dict_viz = eval_viz_uni_chamfer(mesh_pred.vertices, pointcloud_gt, mesh_pred.vertex_normals, normals_gt)
        out_dict.update(out_dict_viz)
    return out_dict


def eval_pointcloud(pointcloud_pred,
                    pointcloud_gt,
                    normals_pred=None,
                    normals_gt=None,
                    return_error_pcs=False,
                    metric_space = True,
                    subject = None,
                    expression = None):

    if not metric_space:
        thresholds = [0.005, 0.01, 0.015, 0.02]
    else:
        thresholds = [1, 5, 10, 20] # scale in mm

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    if metric_space:
        manager = DataManager()
        transform_from_metric = manager.get_transform_from_metric(subject, expression)
        scale_nphm_2_metric = 1/transform_from_metric['s']
        pointcloud_pred *= scale_nphm_2_metric
        pointcloud_gt *= scale_nphm_2_metric


    # Completeness: how far are the points of the target point cloud
    # from the predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    completeness_pc = completeness
    completeness_pc_normals = completeness_normals
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()


    # Accuracy: how far are the points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy_pc = accuracy
    accuracy_pc_normals = accuracy_normals
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()


    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2
    chamfer_l1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]

    if not normals_pred is None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan


    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals consistency': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'chamfer_l1': chamfer_l1,
        'f_score_05': F[0], # threshold: metric: 1mm,  otherwise 0.005
        'f_score_10': F[1], # threshold: metric: 5mm,  otherwise 0.01
        'f_score_15': F[2], # threshold: metric: 10mm, otherwise 0.015
        'f_score_20': F[3], # threshold: metric: 12mm, otherwise 0.020
    }

    if return_error_pcs:
        return out_dict, {'completeness': completeness_pc,
                          'accuracy': accuracy_pc,
                          'completeness_normals': completeness_pc_normals,
                          'accuracy_normals': accuracy_pc_normals}
    else:
        return out_dict


def eval_viz_uni_chamfer(pointcloud_pred, pointcloud_gt,
                    normals_pred=None, normals_gt=None):
    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Accuracy: how far are the points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )

    if normals_gt is None:
        accuracy_normals = np.nan

    out_dict = {
        'dist_pred2gt': accuracy,
        'nsim_pred2gt': accuracy_normals,
    }

    return out_dict


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold