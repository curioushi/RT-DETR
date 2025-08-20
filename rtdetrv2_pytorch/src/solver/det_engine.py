"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable
import os
import json

import cv2
import numpy as np
from datetime import datetime

import torch
import torch.amp 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, epoch, output_dir: str, **kwargs):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    writer = kwargs.get('writer', None)
    
    # Initialize 4x4 grid for visualization
    grid_images = []
    target_image_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        for target in targets:
            target['boxes_xyxy'] = target['boxes']
            x1, y1, x2, y2 = target['boxes_xyxy'][:, 0], target['boxes_xyxy'][:, 1], target['boxes_xyxy'][:, 2], target['boxes_xyxy'][:, 3]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            target['boxes'] = torch.stack([cx, cy, w, h], dim=1) / 640.0

        outputs = model(samples, targets=targets)
        loss_dict = criterion(outputs, targets)
        if writer and dist_utils.is_main_process():
            for k, v in loss_dict.items():
                writer.add_scalar(f'Test/{k}', v.item(), epoch)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        for sample, target, output in zip(samples, targets, results):
            image_id_val = target['image_id'].item()
            if image_id_val not in target_image_ids:
                continue
                
            # 1. Convert sample to OpenCV format (H, W, C), BGR, 0-255
            img_tensor_cpu = sample.cpu().detach() # Should be (C, H, W) and range [0,1]
            img_numpy_chw = img_tensor_cpu.numpy()
            # Transpose C, H, W to H, W, C
            img_numpy_hwc_rgb = np.transpose(img_numpy_chw, (1, 2, 0))
            
            # Scale 0-1 to 0-255 and clip for safety
            # Ensure the image is contiguous in memory after transpose for cvtColor
            img_numpy_hwc_rgb_contiguous = np.ascontiguousarray(img_numpy_hwc_rgb)
            img_to_save = (np.clip(img_numpy_hwc_rgb_contiguous, 0, 1) * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

            img_gt_vis = img_bgr.copy()
            img_pred_vis = img_bgr.copy()

            # 2. Draw Ground Truth boxes
            gt_boxes_tensor = target['boxes_xyxy'] 
            gt_labels_tensor = target['labels']
            gt_quads_tensor = target['coords2d']
            w, h = target['orig_size'].cpu().detach().numpy()

            gt_boxes_np = gt_boxes_tensor.cpu().detach().numpy()
            gt_labels_np = gt_labels_tensor.cpu().detach().numpy()
            gt_quads_np = gt_quads_tensor.cpu().detach().numpy() * 640

            class_to_colors = {
                0: (255, 0, 0),
                1: (0, 255, 0),
                2: (0, 0, 255),
                3: (255, 255, 0),
            }
            class_to_colors.update({ i:(0, 0, 0) for i in range(4, 500)})

            colored_mask = img_gt_vis.copy()
            for i in range(gt_boxes_np.shape[0]):
                label = gt_labels_np[i]
                quads = gt_quads_np[i].reshape(4, 2)
                cv2.polylines(img_gt_vis, [quads.astype(np.int32)], True, class_to_colors[label], 2)

            img_gt_vis = cv2.addWeighted(img_gt_vis, 0.8, colored_mask, 0.2, 0)

            # 3. Draw Prediction boxes (score > 0.6)
            pred_boxes_tensor = output['boxes']
            pred_scores_tensor = output['scores']
            pred_labels_tensor = output['labels']
            pred_quads_tensor = output['quads']

            pred_boxes_np = pred_boxes_tensor.cpu().detach().numpy() / 1024 * 640
            pred_scores_np = pred_scores_tensor.cpu().detach().numpy()
            pred_labels_np = pred_labels_tensor.cpu().detach().numpy()
            pred_quads_np = pred_quads_tensor.cpu().detach().numpy() * 640

            score_thresh = 0.6
            colored_mask = img_pred_vis.copy()
            for i in range(pred_boxes_np.shape[0]):
                if pred_scores_np[i] > score_thresh:
                    label = pred_labels_np[i]
                    quads = pred_quads_np[i].reshape(4, 2)
                    cv2.polylines(img_pred_vis, [quads.astype(np.int32)], True, class_to_colors[label], 2)

            img_pred_vis = cv2.addWeighted(img_pred_vis, 0.8, colored_mask, 0.2, 0)

            # Store images for grid
            grid_images.append((image_id_val, img_gt_vis, img_pred_vis))
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # Create 4x4 grid
    if grid_images:
        # Sort by image_id to ensure correct order
        grid_images.sort(key=lambda x: x[0])
        
        # Get image dimensions
        img_height, img_width = grid_images[0][1].shape[:2]
        
        # Create 4x4 grid
        grid_rows = 4
        grid_cols = 4
        grid_img = np.zeros((img_height * grid_rows, img_width * grid_cols, 3), dtype=np.uint8)
        
        # Fill grid: first row: gt_0, pred_0, gt_1, pred_1, etc.
        for idx, (image_id, gt_img, pred_img) in enumerate(grid_images):
            row = idx // 2  # 0, 0, 1, 1, 2, 2, 3, 3
            col = (idx % 2) * 2 + (image_id % 2)  # 0, 1, 2, 3 for each row
            
            if image_id % 2 == 0:  # Even image_id: gt goes to col 0, pred goes to col 1
                grid_img[row*img_height:(row+1)*img_height, 0*img_width:1*img_width] = gt_img
                grid_img[row*img_height:(row+1)*img_height, 1*img_width:2*img_width] = pred_img
            else:  # Odd image_id: gt goes to col 2, pred goes to col 3
                grid_img[row*img_height:(row+1)*img_height, 2*img_width:3*img_width] = gt_img
                grid_img[row*img_height:(row+1)*img_height, 3*img_width:4*img_width] = pred_img
        
        # Save grid image
        cv2.imwrite(os.path.join(output_dir, f"validate_{epoch:04}.png"), grid_img)
        cv2.imwrite(os.path.join(output_dir, "latest.png"), grid_img)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



