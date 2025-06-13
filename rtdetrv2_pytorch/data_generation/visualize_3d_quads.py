import os
import rerun as rr
import argparse
import numpy as np
import json
from glob import glob

def quad_to_3d(quad, normal, offset, camera_K):
    quad = np.array(quad).reshape(4, 2)
    normal = np.array(normal)
    offset = np.array(offset)

    quad_homo = np.hstack([quad, np.ones((4, 1))])
    ray_dir = (np.linalg.inv(camera_K) @ quad_homo.T).T
    ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=1)[:, None]

    denom = normal @ ray_dir.T
    t = -offset / (denom + 1e-7)
    quad_3d = t[:, None] * ray_dir
    return quad_3d

def make_rerun_quad(center, rotation, size):
    center = np.array(center)
    rotation = np.array(rotation)
    size = np.array(size)
    normal = rotation[:3]
    x_axis = rotation[3:]
    y_axis = np.cross(normal, x_axis)
    normal /= np.linalg.norm(normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    half_x_axis = x_axis * size[0] / 2
    half_y_axis = y_axis * size[1] / 2
    quad_3d = np.array([
        center + half_x_axis + half_y_axis,
        center + half_x_axis - half_y_axis,
        center - half_x_axis - half_y_axis,
        center - half_x_axis + half_y_axis,
    ])
    return quad_3d

def quad_to_mesh(quad_3d_list, color=[255, 255, 255]):
    vertex_positions = []
    vertex_normals = []
    vertex_colors = []
    triangle_indices = []
    for quad_3d in quad_3d_list:
        normal = np.cross((quad_3d[1] - quad_3d[0]), (quad_3d[2] - quad_3d[1]))
        normal /= np.linalg.norm(normal)
        offset = len(vertex_positions)
        triangle_indices.append([offset + 0, offset + 1, offset + 2])
        triangle_indices.append([offset + 0, offset + 2, offset + 3])
        for p in quad_3d:
            vertex_positions.append(p)
            vertex_normals.append(normal)
            vertex_colors.append(color)
    return rr.Mesh3D(
        vertex_positions=vertex_positions,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors,
        triangle_indices=triangle_indices,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_dir", type=str)
    args = parser.parse_args()

    predictions_files = sorted(glob(os.path.join(args.predictions_dir, "*.json")))
    predictions_files = [x for x in predictions_files if "prediction" in x]

    rr.init("quads", spawn=True)
    for i, predictions_file in enumerate(predictions_files):
        with open(predictions_file, "r") as f:
            data = json.load(f)

        ground_truth = data["ground_truth"]
        # # hack
        # fx, cx, fy, cy = np.array(ground_truth["camera_Ks"][0]) / 1024
        # camera_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        predictions = data["predictions"]

        gt_quad_3d_list = []
        for center, rotation, size in zip(
            ground_truth["centers"],
            ground_truth["rotations"],
            ground_truth["sizes"],
        ):
            gt_quad_3d_list.append(make_rerun_quad(center, rotation, size))

        pred_quad_3d_list = []
        for score, center, rotation, size in zip(
            predictions["scores"],
            predictions["centers"],
            predictions["rotations"],
            predictions["sizes"],
        ):
            if score < 0.6:
                continue
            pred_quad_3d_list.append(make_rerun_quad(center, rotation, size))
        

        rr.set_time_sequence("frame", i)
        rr.log(
            "gt_quads",
            quad_to_mesh(gt_quad_3d_list, color=[200, 255, 200]),
        )
        rr.log(
            "pred_quads",
            quad_to_mesh(pred_quad_3d_list, color=[255, 200, 200]),
        )