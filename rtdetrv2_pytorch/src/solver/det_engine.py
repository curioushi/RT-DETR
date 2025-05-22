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


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, output_dir: str):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for sample, target, output in zip(samples, targets, results):
            if target['image_id'].item() == 4:
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
                img_pred_vis_quad = img_bgr.copy()

                image_id_val = target['image_id'].item()

                # 2. Draw Ground Truth boxes
                gt_quads_tensor = target['coords'] 
                gt_weights_tensor = target['weights']
                gt_normals_tensor = gt_weights_tensor[:, :3]
                gt_offsets_tensor = gt_weights_tensor[:, 3]

                gt_quads_np = gt_quads_tensor.cpu().numpy() / 1024 * 640
                gt_normals_np = gt_normals_tensor.cpu().numpy()
                gt_offsets_np = gt_offsets_tensor.cpu().numpy()

                json_data = dict()
                json_data['ground_truth'] = {
                    "camera_Ks": target['camera_Ks'].cpu().numpy().tolist(),
                    "quads": gt_quads_np.tolist(),
                    "normals": gt_normals_np.tolist(),
                    "offsets": gt_offsets_np.tolist(),
                }

                for i in range(gt_quads_np.shape[0]):
                    quad = gt_quads_np[i] # [x1, y1, x2, y2, x3, y3, x4, y4]
                    normal = gt_normals_np[i] # [nx, ny, nz]
                    normal[1] *= -1
                    normal[2] *= -1
                    offset = gt_offsets_np[i]
                    quad_center = np.array(quad).reshape(4, 2).mean(axis=0)

                    # Color from normal: (normal_component + 1) / 2 * 255
                    # Components assumed to be in [-1, 1]
                    color_r = int(((normal[0] + 1) / 2) * 255)
                    color_g = int(((normal[1] + 1) / 2) * 255)
                    color_b = int(((normal[2] + 1) / 2) * 255)
                    draw_color_bgr = (color_b, color_g, color_r) # OpenCV uses BGR

                    # Draw quad
                    cv2.polylines(img_gt_vis, [quad.astype(np.int32).reshape(-1, 2)], True, draw_color_bgr, 1, cv2.LINE_AA)

                    # Draw offset
                    text = f'{offset:.3f}'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_x = int(quad_center[0] - text_width/2)
                    text_y = int(quad_center[1] + text_height/2)
                    cv2.putText(img_gt_vis, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color_bgr, 1, cv2.LINE_AA)

                filename_gt = os.path.join(output_dir, f"image_{image_id_val}_gt.png")
                cv2.imwrite(filename_gt, img_gt_vis)

                # 3. Draw Prediction boxes (score > 0.6)
                pred_boxes_tensor = output['boxes']
                pred_scores_tensor = output['scores']
                pred_quads_tensor = output['quads']
                pred_weights_tensor = output['weights']
                pred_normals_tensor = pred_weights_tensor[:, :3]
                pred_offsets_tensor = pred_weights_tensor[:, 3]

                pred_boxes_np = pred_boxes_tensor.cpu().detach().numpy() / 1024 * 640
                pred_scores_np = pred_scores_tensor.cpu().detach().numpy()
                pred_quads_np = pred_quads_tensor.cpu().detach().numpy() * 640
                pred_normals_np = pred_normals_tensor.cpu().detach().numpy()
                pred_offsets_np = pred_offsets_tensor.cpu().detach().numpy()

                json_data['predictions'] = {
                    "scores": pred_scores_np.tolist(),
                    "quads": pred_quads_np.tolist(),
                    "normals": pred_normals_np.tolist(),
                    "offsets": pred_offsets_np.tolist(),
                }
                with open(os.path.join(output_dir, f"prediction_{image_id_val}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
                    json.dump(json_data, f, indent=2)

                score_thresh = 0.6
                for i in range(pred_quads_np.shape[0]):
                    if pred_scores_np[i] > score_thresh:
                        box = pred_boxes_np[i]
                        quad = pred_quads_np[i] # [x1, y1, x2, y2, x3, y3, x4, y4]
                        normal = pred_normals_np[i] # [nx, ny, nz]
                        normal[1] *= -1
                        normal[2] *= -1
                        offset = pred_offsets_np[i]
                        quad_center = np.array(quad).reshape(4, 2).mean(axis=0)

                        color_r = int(((normal[0] + 1) / 2) * 255)
                        color_g = int(((normal[1] + 1) / 2) * 255)
                        color_b = int(((normal[2] + 1) / 2) * 255)
                        draw_color_bgr = (color_b, color_g, color_r) # OpenCV uses BGR

                        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(img_pred_vis, (xmin, ymin), (xmax, ymax), draw_color_bgr, 2)

                        cv2.polylines(img_pred_vis_quad, [quad.astype(np.int32).reshape(-1, 2)], True, draw_color_bgr, 1, cv2.LINE_AA)

                        # Draw offset
                        text = f'{offset:.3f}'
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_x = int(quad_center[0] - text_width/2)
                        text_y = int(quad_center[1] + text_height/2)
                        cv2.putText(img_pred_vis_quad, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color_bgr, 1, cv2.LINE_AA)

                
                filename_pred_quad = os.path.join(output_dir, f"image_{image_id_val}_pred_quad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(filename_pred_quad, img_pred_vis_quad)
                filename_pred_box = os.path.join(output_dir, f"image_{image_id_val}_pred_box_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(filename_pred_box, img_pred_vis)
        if coco_evaluator is not None:
            coco_evaluator.update(res)

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



