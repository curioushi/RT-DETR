import os
import json
import cv2
import imageio.v2 as imageio
import numpy as np
import os.path as osp
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from glob import glob

def pos_quat_to_mat44(pos, quat):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def draw_box(color_image, camera_K, tf_cam_box, size, color=None, linewidth=1):
    line_indices = [
        [0,1],
        [1,2],
        [2,3],
        [3,0],
        [4,5],
        [5,6],
        [6,7],
        [7,4],
        [0,4],
        [1,5],
        [2,6],
        [3,7],
    ]
    length, width, height = size
    corners = np.array([
        [-length/2, -width/2, -height/2],
        [length/2, -width/2, -height/2],
        [length/2, width/2, -height/2],
        [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2],
        [length/2, -width/2, height/2],
        [length/2, width/2, height/2],
        [-length/2, width/2, height/2],
    ])
    corners_in_camera = (tf_cam_box[:3, :3] @ corners.T + tf_cam_box[:3, 3][:, None]).T  # N, 3
    
    # project to image plane
    uv = camera_K @ corners_in_camera.T  # 3, N
    uv = uv[:2] / uv[2]  # 2, N
    uv = uv.T  # N, 2
    uv_int = uv.astype(np.int32)

    # draw lines
    if color is None:
        color = np.random.randint(0, 255, 3).tolist()
    for i, j in line_indices:
        if corners_in_camera[i, 2] > 0 and corners_in_camera[j, 2] > 0:
            cv2.line(color_image, uv_int[i], uv_int[j], color, linewidth)
        elif corners_in_camera[i, 2] < 0 and corners_in_camera[j, 2] > 0:
            p_i = corners_in_camera[i]
            p_j = corners_in_camera[j]
            t = (-p_i[2] + 0.01) / (p_j[2] - p_i[2])
            p_int = p_i + t * (p_j - p_i)
            uv3 = camera_K @ p_int
            uv_mid = uv3[:2] / uv3[2]
            uv_mid = uv_mid.astype(np.int32)
            cv2.line(color_image, tuple(uv_mid), tuple(uv_int[j]), color, linewidth)
        elif corners_in_camera[i, 2] > 0 and corners_in_camera[j, 2] < 0:
            p_i = corners_in_camera[i]
            p_j = corners_in_camera[j]
            t = (-p_j[2] + 0.01) / (p_i[2] - p_j[2])
            p_int = p_j + t * (p_i - p_j)
            uv3 = camera_K @ p_int
            uv_mid = uv3[:2] / uv3[2]
            uv_mid = uv_mid.astype(np.int32)
            cv2.line(color_image, tuple(uv_int[i]), tuple(uv_mid), color, linewidth)
        
def extract_box_rect(mask):
    rows, cols = np.where(mask)
    if rows.size > 0 and cols.size > 0:
        xmin = np.min(cols)
        xmax = np.max(cols)
        ymin = np.min(rows)
        ymax = np.max(rows)
        
        x = int(xmin)
        y = int(ymin)
        width = int(xmax - xmin + 1)
        height = int(ymax - ymin + 1)
        
        rect = (x, y, width, height) 
    else:
        rect = None 
    return rect

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

def save_pcd(points, filepath):
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {points.shape[0]}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {points.shape[0]}
DATA ascii
"""
    with open(filepath, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%f %f %f")
    print(f"Saved point cloud to {filepath}")

def compute_face_support_vector(normalized_points):
    face_support_vector = np.zeros(6)
    for axis in [0, 1, 2]:
        for i, offset in enumerate([-0.5, 0.5]):
            plane_mask = np.abs(normalized_points[:, axis] - offset) < 0.001
            plane_points = normalized_points[plane_mask]
            current_support_vector = np.zeros(6)
            if plane_points.shape[0] > 0:
                current_support_vector[axis * 2 + i] = 1
                j = (axis + 1) % 3
                k = (axis + 2) % 3
                points_j = plane_points[:, j]
                min_j, max_j = np.min(points_j), np.max(points_j)
                points_k = plane_points[:, k]
                min_k, max_k = np.min(points_k), np.max(points_k)

                current_support_vector[j * 2] = 0.5 - min_j
                current_support_vector[j * 2 + 1] = 0.5 + max_j
                current_support_vector[k * 2] = 0.5 - min_k
                current_support_vector[k * 2 + 1] = 0.5 + max_k
                current_support_vector = np.clip(current_support_vector, 0, 1)
            face_support_vector = np.maximum(face_support_vector, current_support_vector)
    return face_support_vector

def shrink_box(tf_box, box_size, face_support_vector):
    shrink_ratio = 1 - face_support_vector
    x0, x1 = -box_size[0] / 2, box_size[0] / 2
    y0, y1 = -box_size[1] / 2, box_size[1] / 2
    z0, z1 = -box_size[2] / 2, box_size[2] / 2
    x0 += shrink_ratio[0] * box_size[0]
    x1 -= shrink_ratio[1] * box_size[0]
    y0 += shrink_ratio[2] * box_size[1]
    y1 -= shrink_ratio[3] * box_size[1]
    z0 += shrink_ratio[4] * box_size[2]
    z1 -= shrink_ratio[5] * box_size[2]
    new_box_size = np.array([x1 - x0, y1 - y0, z1 - z0])
    offset = (np.array([x0, y0, z0]) + np.array([x1, y1, z1])) / 2
    tf_offset = np.eye(4)
    tf_offset[:3, 3] = offset
    new_tf_box = tf_box @ tf_offset
    return new_tf_box, new_box_size
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_root", type=str)
    args = parser.parse_args()

    pattern = os.path.join(args.dataset_root, "[0-9][0-9][0-9][0-9]")
    scene_dirs = sorted([d for d in glob(pattern) if os.path.isdir(d)])

    for scene_dir in scene_dirs:
        scene_name = osp.basename(scene_dir)

        color_image = imageio.imread(osp.join(scene_dir, "color.png"))
        instmap = imageio.imread(osp.join(scene_dir, "instmap.tiff"))
        depth = imageio.imread(osp.join(scene_dir, "depth.tiff"))
        scene_json = json.load(open(osp.join(scene_dir, "scene.json")))
        camera_K = np.array(scene_json["camera"]["intrinsics"])
        xyz_map = compute_xyz_map(depth, camera_K)

        tf_world_cam = pos_quat_to_mat44(scene_json["camera"]["position"], scene_json["camera"]["rotation"])
        
        color_before_shrink = color_image.copy()
        color_after_shrink = color_image.copy()
        for i, box in enumerate(scene_json["boxes"]):
            if not box["visible"]:
                continue
            box_size = box["size"]
            tf_world_box = pos_quat_to_mat44(box["position"], box["rotation"])
            tf_cam_box = np.linalg.inv(tf_world_cam) @ tf_world_box
            box_mask = (instmap == i)
            box_points = xyz_map[box_mask]
            tf_box_cam = np.linalg.inv(tf_cam_box)
            box_points = (tf_box_cam[:3, :3] @ box_points.T + tf_box_cam[:3, 3][:, None]).T
            box_points /= np.array(box_size)[None, :]
            face_support_vector = compute_face_support_vector(box_points)
            # rect = extract_box_rect(box_mask)
            # if rect is not None:
            #     x, y, width, height = rect
            #     cv2.rectangle(color_image, (x, y), (x + width, y + height), np.random.randint(0, 255, 3).tolist(), 2)

            #     text = ",".join([f"{v:.1f}" for v in face_support_vector])
            #     text = f'({text})'
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.4
            #     color = (0, 255, 0)
            #     thickness = 1
            #     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            #     cv2.putText(color_image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            if face_support_vector.sum() < 4.5:
                continue
            draw_box(color_before_shrink, camera_K, tf_cam_box, box_size, color=(0, 255, 0), linewidth=2)
            tf_cam_box, box_size = shrink_box(tf_cam_box, box["size"], face_support_vector)
            draw_box(color_after_shrink, camera_K, tf_cam_box, box_size, color=(0, 255, 0), linewidth=2)
        cv2.imwrite(osp.join(scene_dir, "color1.png"), color_before_shrink)
        cv2.imwrite(osp.join(scene_dir, "color2.png"), color_after_shrink)
        
        # tf_world_container = pos_quat_to_mat44(scene_json["container"]["position"], scene_json["container"]["rotation"])
        # length, width, height = scene_json["container"]["size"]
        # x_axis = tf_world_container[:3, 0]
        # z_axis = tf_world_container[:3, 2]
        # tf_world_container[:3, 3] += z_axis * height / 2
        # tf_world_container[:3, 3] -= x_axis * length / 2
        # tf_cam_container = np.linalg.inv(tf_world_cam) @ tf_world_container
        # draw_box(color_image, camera_K, tf_cam_container, scene_json["container"]["size"], (0, 255, 0))
            

        # cv2.imshow("color", color_image)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
