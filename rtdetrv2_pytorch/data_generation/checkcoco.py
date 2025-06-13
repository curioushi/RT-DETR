import os
import cv2
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from pycocotools import mask as maskUtils
import rerun as rr

def project_3d_to_2d(center, rotation, size, camera_K):
    normal = rotation[:3]
    x_axis = rotation[3:]
    y_axis = np.cross(normal, x_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    half_x_axis = x_axis * size[0] * 0.5
    half_y_axis = y_axis * size[1] * 0.5
    quads = np.array([
        center + half_x_axis + half_y_axis,
        center + half_x_axis - half_y_axis,
        center - half_x_axis - half_y_axis,
        center - half_x_axis + half_y_axis,
    ])
    quads = quads @ camera_K.T
    quads = quads[:, :2] / quads[:, 2:3]
    return quads

def project_quad2d_to_plane(quad_2d, normal, offset, camera_K):
    quad_2d_homo = np.concatenate([quad_2d, np.ones((quad_2d.shape[0], 1))], axis=1)
    quad_2d_homo = quad_2d_homo @ np.linalg.inv(camera_K).T  # 4 x 3
    t = -offset / (quad_2d_homo @ normal)
    quad_3d = quad_2d_homo * t[:, None]
    return quad_3d

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

def normalize_depth(depth):
    min_depth, max_depth = depth.min(), depth.max()
    return (depth - min_depth) / (max_depth - min_depth)

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

def unproject_depth(depth, camera_K):
    fx, cx, fy, cy = camera_K[0, 0], camera_K[0, 2], camera_K[1, 1], camera_K[1, 2]
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return np.stack([x * depth, y * depth, depth], axis=2)

def check_coco_annotations(imgs_dir, anno_json, pred_dir=None):
    """
    Check COCO format annotation file and visualize
    """
    # Load annotation file
    with open(anno_json, 'r') as f:
        coco_data = json.load(f)
    
    # Load prediction file
    file_name_to_predictions = {}
    if pred_dir is not None:
        pred_files = sorted(glob(os.path.join(pred_dir, "*.npz")))
        for pred_file in pred_files:
            file_name = os.path.basename(pred_file).split('_')[-1].replace(".npz", ".png")
            file_name_to_predictions[file_name] = pred_file
    
    # Print key fields of COCO data
    print("\n===== COCO Dataset Information =====")
    print(f"Description: {coco_data.get('info', {}).get('description', 'No description')}")
    print(f"Number of images: {len(coco_data.get('images', []))}")
    print(f"Number of annotations: {len(coco_data.get('annotations', []))}")
    print(f"Number of categories: {len(coco_data.get('categories', []))}")
    
    # Print category information
    print("\n===== Category Information =====")
    for category in coco_data.get('categories', []):
        print(f"ID: {category['id']}, Name: {category['name']}, Supercategory: {category.get('supercategory', 'None')}")
    
    # Create mapping from image ID to filename
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
    
    # Create mapping from image ID to annotations
    image_id_to_annotations = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # Create mapping from category ID to name
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Visualize each image and its annotations
    print("\n===== Starting Visualization of Images and Annotations =====")
    
    bbox_counter = Counter()
    rr.init("quads", spawn=True)
    for image_id, file_name in tqdm(image_id_to_file.items(), desc="Processing images"):
        # Load image
        img_path = os.path.join(imgs_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image file does not exist: {img_path}")
            continue
        
        npz_path = os.path.join(imgs_dir, "..", "xyz", file_name.replace(".png", ".npz"))
        if not os.path.exists(npz_path):
            print(f"Warning: XYZ file does not exist: {npz_path}")
            continue
        gt_xyz = np.load(npz_path)['xyz_mapping']
        gt_depth = gt_xyz[:, :, 2]
    
        # Load prediction
        if pred_dir is not None:
            pred_file = file_name_to_predictions.get(file_name, [])
            predictions = np.load(pred_file)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image: {img_path}")
            continue
        
        pred_mask_size = (160, 160)
        rr.set_time_sequence("frame", image_id)
        # Draw directly on the original image
        # Draw all annotations
        annotations = image_id_to_annotations.get(image_id, [])
        gt_quad_img = img.copy()
        gt_bbox_img = img.copy()
        gt_mask_img = img.copy()
        gt_quad_3d_list = []
        camera_K = None
        for ann in annotations:
            # Get bounding box
            x, y, w, h = map(int, ann['bbox'])
            category_id = ann['category_id']
            category_name = category_id_to_name.get(category_id, f"Unknown category {category_id}")
            # color = colors[category_id]
            color = (0, 255, 0)
            if camera_K is None and 'camera_K' in ann:
                camera_K = np.array(ann['camera_K'])
            
            # Draw bounding box on the original image
            if ann['pose'] is not None:
                normal = np.array(ann['pose'])[:3, 2]
                normal[1:] *= -1  # rotate x 180 to make z axis out of the screen
                color = [int((c + 1) / 2 * 255) for c in normal][::-1] # rgb -> bgr

            if ann['coords2d'] is not None:
                coords = ann['coords2d']  # 4 x 2
                quad_center = np.mean(coords, axis=0)
                quad_edge_mp = np.mean(coords[:2], axis=0)
                cv2.polylines(gt_quad_img, [np.array(coords).astype(int)], True, color, 1, lineType=cv2.LINE_AA)
                cv2.line(gt_quad_img, quad_center.astype(int), quad_edge_mp.astype(int), color, 1, lineType=cv2.LINE_AA)
            
            if ann['segmentation'] is not None:
                mask = maskUtils.decode(ann['segmentation'])
                gt_mask_img[mask > 0.5] = color
            
            if 'pose' in ann and 'size' in ann:
                pose = np.array(ann['pose'])
                size = np.array(ann['size'])
                center = pose[:3, 3]
                normal = pose[:3, 2]
                x_axis = pose[:3, 0]
                quad_3d = make_rerun_quad(center, np.concatenate([normal, x_axis]), size)
                gt_quad_3d_list.append(quad_3d)

            cv2.rectangle(gt_bbox_img, (x, y), (x + w, y + h), color, 1)

        rr.log(
            "gt_quads",
            quad_to_mesh(gt_quad_3d_list, color=[200, 255, 200]),
        )
        # rr.log(
        #     "gt_xyz",
        #     rr.Points3D(positions=gt_xyz.reshape(-1, 3), colors=[200, 255, 200], radii=0.001)
        # )
        num_points = 8192
        row_indices = np.random.randint(0, gt_xyz.shape[0], num_points)
        col_indices = np.random.randint(0, gt_xyz.shape[1], num_points)
        gt_xyz_sparse = gt_xyz[row_indices, col_indices, :]
        rr.log(
            "gt_xyz_sparse",
            rr.Points3D(positions=gt_xyz_sparse, colors=[200, 255, 200], radii=0.005)
        )
        bbox_counter[len(annotations)] += 1
        
        if pred_dir is not None:
            boxes = predictions['boxes']
            quads = predictions['quads'] * 1024
            labels = predictions['labels']
            scores = predictions['scores']
            normals = predictions['normals']
            offsets = predictions['offsets']
            masks = sigmoid_np(predictions['masks'])

            pred_mask_img = cv2.resize(img, pred_mask_size, interpolation=cv2.INTER_NEAREST)
            pred_quad_img = img.copy()
            pred_bbox_img = img.copy()
            pred_quad_3d_list = []
            for box, label, score, quad, normal, offset, mask in zip(boxes, labels, scores, quads, normals, offsets, masks):
                quad = quad.reshape(-1, 2)
                quad_center = np.mean(quad, axis=0)
                quad_edge_mp = np.mean(quad[:2], axis=0)
                quad_3d = project_quad2d_to_plane(quad, normal, offset, camera_K)
                if score < 0.6:
                    continue
                if normal is not None:
                    normal = np.array(normal)
                    normal = normal / np.linalg.norm(normal)
                    normal[1:] *= -1  # rotate x 180 to make z axis out of the screen
                    color = [int((np.clip(c, -1, 1) + 1) / 2 * 255) for c in normal][::-1]
                else:
                    color = np.random.randint(0, 255, 3).tolist()
                
                # projected_quads = project_3d_to_2d(center, rotation, size, camera_K)
                cv2.polylines(pred_quad_img, [np.array(quad).astype(int)], True, color, 1, lineType=cv2.LINE_AA)
                cv2.line(pred_quad_img, quad_center.astype(int), quad_edge_mp.astype(int), color, 1, lineType=cv2.LINE_AA)

                pred_mask_img[mask > 0.5] = color

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(pred_bbox_img, (x1, y1), (x2, y2), color, 1)

                pred_quad_3d_list.append(quad_3d)

            quad_img = np.hstack([gt_quad_img, pred_quad_img])
            bbox_img = np.hstack([gt_bbox_img, pred_bbox_img])
            mask_img = np.hstack([gt_mask_img, 
                                  cv2.resize(pred_mask_img, gt_mask_img.shape[:2], interpolation=cv2.INTER_NEAREST)])
            depth_img = (np.hstack([normalize_depth(gt_depth), normalize_depth(gt_depth)]) * 255).astype(np.uint8)
            depth_img = np.stack([depth_img, depth_img, depth_img], axis=2)
            compare_img = np.hstack([np.vstack([quad_img, bbox_img]), np.vstack([mask_img, depth_img])])
            cv2.imwrite(f"{image_id:04}_compare.jpg", compare_img)

        rr.log(
            "pred_quads",
            quad_to_mesh(pred_quad_3d_list, color=[255, 200, 200]),
        )


    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.bar(bbox_counter.keys(), bbox_counter.values())
    plt.xlabel('Number of bounding boxes')
    plt.ylabel('Number of images')
    plt.title('Bounding box distribution')
    plt.savefig('bbox_distribution.png')
    plt.close()
    print("Visualization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check COCO format annotation file and visualize")
    parser.add_argument("imgs_dir", type=str, help="Path to images directory")
    parser.add_argument("anno_json", type=str, help="Path to COCO format annotation file")
    parser.add_argument("--pred_dir", type=str, help="Optional: Path to prediction directory", default=None)
    
    args = parser.parse_args()
    
    check_coco_annotations(args.imgs_dir, args.anno_json, pred_dir=args.pred_dir)
