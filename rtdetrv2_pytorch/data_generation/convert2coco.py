import os
import json
import argparse
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
from datetime import datetime
import shutil
from utils import xyzquat_to_mat44, compute_xyz_map, compute_support_vector, flip_rotate_coords, estimate_quad3d_pose_size
from pycocotools import mask as maskUtils

def convert_to_coco_core(scene_dirs, output_dir):
    images_dir = os.path.join(output_dir, "images")
    xyz_dir = os.path.join(output_dir, "xyz")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)

    # Initialize COCO format data
    coco_data = {
        "info": {
            "description": "Converted from BlenderProc generated data",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Haoqi Shi",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "No known copyright restrictions",
                "url": "",
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "carton",
                "supercategory": "box"
            },
            {
                "id": 2,
                "name": "container",
                "supercategory": "box"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    
    annotation_id = 1
    
    # Process each scene
    for image_id, scene_dir in enumerate(tqdm(scene_dirs, desc="Processing scenes")):
        scene_name = os.path.basename(scene_dir)
        
        # Read image and instance segmentation map
        color_image_path = os.path.join(scene_dir, "color.png")
        scene_json_path = os.path.join(scene_dir, "scene.json")
        quad_instmap_path = os.path.join(scene_dir, "quad_instmap.npz")
        depth_path = os.path.join(scene_dir, "depth.npz")
        
        if not (os.path.exists(color_image_path) and 
                os.path.exists(scene_json_path) and 
                os.path.exists(quad_instmap_path) and 
                os.path.exists(depth_path)):
            print(f"Warning: Scene {scene_name} is missing required files, skipping")
            continue

        # Read scene information
        with open(scene_json_path, "r") as f:
            scene_data = json.load(f)
        
        # Read images
        color_image = imageio.imread(color_image_path)
        quad_instmap = np.load(quad_instmap_path)["arr_0"]
        depth = np.load(depth_path)["arr_0"]
        camera_K = np.array(scene_data["camera"]["intrinsics"])
        xyz_mapping = compute_xyz_map(depth, camera_K)
        
        box_instance_to_category = dict()
        for box in scene_data["boxes"]:
            box_instance_to_category[box["instance_id"]] = box["category_id"]
        
        # Copy image to output directory
        output_image_path = os.path.join(images_dir, f"{scene_name}.png")
        output_xyz_path = os.path.join(xyz_dir, f"{scene_name}.npz")
        shutil.copy(color_image_path, output_image_path)
        np.savez_compressed(output_xyz_path, xyz_mapping=xyz_mapping.astype(np.float32))
        
        # Add image information
        height, width = color_image.shape[:2]
        coco_data["images"].append({
            "id": image_id,
            "file_name": f"{scene_name}.png",
            "width": width,
            "height": height,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        })

        camera_pose = xyzquat_to_mat44(scene_data["camera"]["position"] + scene_data["camera"]["rotation"])
        
        # Process each visible box
        for quad in scene_data["quads"]:
            if quad['area'] < 900:
                continue
            
            # # Get instance mask
            # mask = (instmap == quad['instance_id'])
            # if not np.any(mask):
            #     continue
            
            # # Calculate segmentation mask
            # rle_segments = mask_to_rle(mask)
            
            # Add annotation information
            quad_pose = xyzquat_to_mat44(quad['position'] + quad['rotation'])
            quad_pose = np.linalg.inv(camera_pose) @ quad_pose
            quad_pose_inv = np.linalg.inv(quad_pose)
            normal = np.array(quad_pose)[:3, 2]
            position = np.array(quad_pose)[:3, 3]
            position_dir = position / np.linalg.norm(position)
            if np.dot(normal, position_dir) > 0:
                normal = -normal
            if np.abs(np.dot(normal, position_dir)) < np.cos(80 / 180 * np.pi):
                continue

            quad_size = quad['size']
            quad_mask = quad_instmap == quad['instance_id']
            
            # Convert quad_mask to RLE format using pycocotools
            rle = maskUtils.encode(np.asfortranarray(quad_mask.astype(np.uint8)))
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')

            quad_points3d = xyz_mapping[quad_mask]
            quad_points3d = (quad_points3d @ quad_pose_inv[:3, :3].T) + quad_pose_inv[:3, 3].reshape(1, 3)
            support_vector, bbox_xxyy = compute_support_vector(quad_points3d, quad_size)
            x0, x1, y0, y1 = bbox_xxyy
            quad_points3d = np.array([
                [x0, y0, 0],
                [x1, y0, 0],
                [x1, y1, 0],
                [x0, y1, 0],
            ])
            quad_points3d = quad_points3d @ quad_pose[:3, :3].T + quad_pose[:3, 3].reshape(1, 3)  # N x 3
            quad_points2d = quad_points3d @ camera_K.T
            assert quad_points2d[:, 2:3].min() > 0
            quad_points2d = quad_points2d[:, :2] / quad_points2d[:, 2:3]
            quad_points2d, quad_points3d = flip_rotate_coords(quad_points2d, quad_points3d)
            quad_pose, quad_size = estimate_quad3d_pose_size(quad_points3d)
            normal = np.array(quad_pose)[:3, 2]
            center = np.array(quad_pose)[:3, 3]
            center_dir = center / np.linalg.norm(center)
            assert np.dot(normal, center_dir) < 0
            assert np.abs(np.dot(normal, center_dir)) > np.cos(80 / 180 * np.pi)

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": box_instance_to_category[quad['parent_instance_id']],
                "bbox": quad['bbox'],
                "area": quad['area'],
                "pose": quad_pose.tolist(),
                "size": quad_size,
                "camera_K": camera_K.tolist(),
                "coords2d": quad_points2d.tolist(),
                "coords3d": quad_points3d.tolist(),
                "segmentation": rle,
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    # Save COCO format data
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Conversion complete! Processed {len(coco_data['images'])} images, generated {len(coco_data['annotations'])} annotations")

def convert_to_coco(input_dir, output_dir, val_ratio):
    """
    Convert data in the input directory to COCO format
    """
    # Create output directory
    output_train_dir = os.path.join(output_dir, "train")
    output_val_dir = os.path.join(output_dir, "val")

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)

    # Find all scene directories
    scene_dirs = sorted(glob(os.path.join(input_dir, "[0-9][0-9][0-9][0-9]")))
    split_idx = int(len(scene_dirs) * (1 - val_ratio))
    train_scene_dirs = scene_dirs[:split_idx]
    val_scene_dirs = scene_dirs[split_idx:]

    convert_to_coco_core(train_scene_dirs, output_train_dir)
    convert_to_coco_core(val_scene_dirs, output_val_dir)
    

def mask_to_rle(binary_mask):
    """
    Convert binary mask to COCO RLE format
    """
    # Find contours in the mask
    contours = []
    mask = binary_mask.astype(np.uint8)
    
    # Simple implementation: find continuous segments in each row
    for i in range(mask.shape[0]):
        row = mask[i]
        runs = []
        start = -1
        
        for j in range(row.shape[0]):
            if row[j] and start == -1:
                start = j
            elif not row[j] and start != -1:
                runs.append([start, j-1])
                start = -1
        
        if start != -1:
            runs.append([start, row.shape[0]-1])
        
        for run in runs:
            contours.append([i, run[0], i, run[1]])
    
    # Convert to COCO format segmentation
    segmentation = []
    for contour in contours:
        segmentation.append([float(x) for x in contour])
    
    return segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BlenderProc generated data to COCO format")
    parser.add_argument("input_dir", type=str, help="Input directory path")
    parser.add_argument("output_dir", type=str, help="Output directory path")
    parser.add_argument("--val_ratio", type=float, help="Validation ratio", default=0.1)
    
    args = parser.parse_args()
    
    convert_to_coco(args.input_dir, args.output_dir, args.val_ratio)
