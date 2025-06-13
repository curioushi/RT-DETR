import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation as R

def colorize_segmap(segmap):
    assert segmap.ndim == 2
    h, w = segmap.shape
    num_classes = int(np.max(segmap))
    variants_per_channel = 2
    while variants_per_channel**3 <= num_classes:
        variants_per_channel += 1
    variations = np.linspace(100, 255, variants_per_channel, endpoint=True).astype(np.uint8)
    colormap = product(variations, variations, variations)
    colormap = np.array(list(colormap))
    color_segmap = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            seg_id = segmap[i, j]
            color_segmap[i, j] = colormap[seg_id]

    return color_segmap

def colorize_depth(depth_image):
    depth_image = depth_image.astype(np.float32)
    valid_depth_mask = depth_image < 100.0
    valid_depth = depth_image[valid_depth_mask]
    min_z, max_z = valid_depth.min(), valid_depth.max()
    depth_image = (depth_image - min_z) / (max_z - min_z)
    depth_image[~valid_depth_mask] = 1.0
    depth_image = (depth_image * 255).astype(np.uint8)
    depth_image = 255 - depth_image
    return depth_image

def colorize_normal(normal_image):
    normal_image = normal_image.astype(np.float32)
    normal_image = (normal_image + 1.0) / 2.0
    normal_image = (normal_image * 255).astype(np.uint8)
    return normal_image

def mat44_to_xyzquat(mat44):
    position = mat44[:3, 3]
    rotation = R.from_matrix(mat44[:3, :3])
    quat = rotation.as_quat()
    return position.tolist() + quat.tolist()

def xyzquat_to_mat44(xyzquat):
    position = np.array(xyzquat[:3])
    rotation = R.from_quat(xyzquat[3:])
    mat44 = np.eye(4)
    mat44[:3, :3] = rotation.as_matrix()
    mat44[:3, 3] = position
    return mat44

def compute_xyz_map(depth, camera_K):
    H, W = depth.shape
    fx = camera_K[0, 0]
    fy = camera_K[1, 1]
    cx = camera_K[0, 2]
    cy = camera_K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz_map = np.stack([X, Y, Z], axis=-1)
    return xyz_map

def mask_to_bbox(mask):
    y_indices, x_indices = np.where(mask)
    x_min = int(np.min(x_indices))
    y_min = int(np.min(y_indices))
    x_max = int(np.max(x_indices))
    y_max = int(np.max(y_indices))
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

def compute_support_vector(points, size):
    """
    Compute the support vector of a box

    Args:
        points: N x 3
        size: (length, width)
    
    Returns:
        support_vector: (x0, x1, y0, y1)    # xxyy format
        bbox: (min_x, max_x, min_y, max_y)  # xxyy format
    """
    x0, x1 = -size[0] / 2, size[0] / 2
    y0, y1 = -size[1] / 2, size[1] / 2
    z = points[:, 2]
    inplane_mask = np.abs(z) < 0.001
    inplane_points = points[inplane_mask]

    min_x, min_y, _ = inplane_points.min(axis=0)
    max_x, max_y, _ = inplane_points.max(axis=0)
    min_x = np.clip(min_x, x0, x1)
    max_x = np.clip(max_x, x0, x1)
    min_y = np.clip(min_y, y0, y1)
    max_y = np.clip(max_y, y0, y1)

    gap_x0 = min_x - x0
    gap_x1 = x1 - max_x
    gap_y0 = min_y - y0
    gap_y1 = y1 - max_y

    shrink_x0 = gap_x0 / size[0]
    shrink_x1 = gap_x1 / size[0]
    shrink_y0 = gap_y0 / size[1]
    shrink_y1 = gap_y1 / size[1]
    shrink_ratio = np.array([shrink_x0, shrink_x1, shrink_y0, shrink_y1])

    return 1 - shrink_ratio, np.array([min_x, max_x, min_y, max_y])

def flip_rotate_coords(points2d, points3d):
    """
    Flip and rotate coordinates
    """
    def main_direction_x(a, b, c, d):
        mid0 = (a + d) / 2
        mid1 = (b + c) / 2
        direction = mid1 - mid0
        direction = direction / np.linalg.norm(direction)
        return direction[0]

    v01 = points2d[1] - points2d[0]
    v12 = points2d[2] - points2d[1]
    cross_product = np.cross(v01, v12)
    if cross_product > 0:
        points2d = points2d[::-1]
        points3d = points3d[::-1]
    
    perms = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ])
    
    main_direction_x_values = [main_direction_x(points2d[perm[0]], 
                                                points2d[perm[1]], 
                                                points2d[perm[2]], 
                                                points2d[perm[3]]) for perm in perms]
    max_index = np.argmin(main_direction_x_values)
    perm = perms[max_index]

    return points2d[perm], points3d[perm]


def estimate_quad3d_pose_size(points3d):
    """
    Estimate the pose (4x4 matrix) and size (length, width) of a 3D rectangle.

    Args:
        points3d (np.ndarray): A 4x3 array of 3D points representing the corners of the rectangle.

    Returns:
        tuple: (pose, size)
            - pose (np.ndarray): A 4x4 transformation matrix.
            - size (list[float]): A list containing [length, width].
    """
    points3d = np.asarray(points3d)
    # Compute center
    center = np.mean(points3d, axis=0)
    
    # Define x_axis
    x_candidate_vec = points3d[1] - points3d[0]
    norm_x_candidate = np.linalg.norm(x_candidate_vec)
    x_axis = x_candidate_vec / norm_x_candidate

    # Define z_axis
    z_axis_candidate = np.cross(x_axis, points3d[2] - points3d[1])
    norm_z_candidate = np.linalg.norm(z_axis_candidate)
    z_axis = z_axis_candidate / norm_z_candidate

    # Define y_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Construct the pose matrix
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = center

    # Estimate size (length, width) by projecting points onto the new axes
    points_centered = points3d - center
    coords_x = points_centered @ x_axis
    coords_y = points_centered @ y_axis
    
    length = np.max(coords_x) - np.min(coords_x)
    width = np.max(coords_y) - np.min(coords_y)
    
    size = [length, width]

    return pose, size