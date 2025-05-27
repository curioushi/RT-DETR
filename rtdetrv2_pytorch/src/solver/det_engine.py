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
                # img_pred_vis_quad = img_bgr.copy()

                image_id_val = target['image_id'].item()

                # 2. Draw Ground Truth boxes
                gt_boxes_tensor = target['boxes'] 
                gt_labels_tensor = target['labels']
                gt_masks_tensor = target['masks']
                gt_depth_tensor = (torch.exp(target['depth']) - 1).squeeze(0)

                gt_boxes_np = gt_boxes_tensor.cpu().detach().numpy()
                gt_labels_np = gt_labels_tensor.cpu().detach().numpy()
                gt_masks_np = gt_masks_tensor.cpu().detach().numpy()
                gt_depth_np = gt_depth_tensor.cpu().detach().numpy()
                min_depth, max_depth = gt_depth_np.min(), gt_depth_np.max()
                gt_depth_np = ((gt_depth_np - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

                class_to_colors = {
                    0: (0, 255, 0),
                    1: (255, 0, 0),
                }
                class_to_colors.update({ i:(0, 0, 0) for i in range(2, 500)})

                colored_mask = img_gt_vis.copy()
                for i in range(gt_boxes_np.shape[0]):
                    x1, y1, x2, y2 = gt_boxes_np[i].astype(np.int32)
                    label = gt_labels_np[i]
                    mask = gt_masks_np[i] > 0.5

                    cv2.rectangle(img_gt_vis, (x1, y1), (x2, y2), class_to_colors[label], 1)
                    colored_mask[mask] = np.random.randint(0, 255, 3)
                img_gt_vis = cv2.addWeighted(img_gt_vis, 0.8, colored_mask, 0.2, 0)

                filename_gt = os.path.join(output_dir, f"image_{image_id_val}_gt.png")
                cv2.imwrite(filename_gt, img_gt_vis)
                filename_gt_depth = os.path.join(output_dir, f"image_{image_id_val}_gt_depth.png")
                cv2.imwrite(filename_gt_depth, gt_depth_np)

                # 3. Draw Prediction boxes (score > 0.6)
                pred_boxes_tensor = output['boxes']
                pred_scores_tensor = output['scores']
                pred_labels_tensor = output['labels']
                pred_masks_tensor = output['masks']
                pred_masks_tensor = F.interpolate(pred_masks_tensor.unsqueeze(1), size=(640, 640), mode='bilinear', align_corners=False).squeeze(1)
                pred_masks_tensor = F.sigmoid(pred_masks_tensor)
                pred_depth_tensor = output['depth']
                pred_depth_tensor = torch.exp(pred_depth_tensor) - 1
                pred_depth_tensor = F.interpolate(pred_depth_tensor.unsqueeze(0).unsqueeze(0), size=(640, 640), mode='nearest').squeeze(0).squeeze(0)

                pred_boxes_np = pred_boxes_tensor.cpu().detach().numpy() / 1024 * 640
                pred_scores_np = pred_scores_tensor.cpu().detach().numpy()
                pred_labels_np = pred_labels_tensor.cpu().detach().numpy()
                pred_masks_np = pred_masks_tensor.cpu().detach().numpy()
                pred_depth_np = pred_depth_tensor.cpu().detach().numpy()
                min_depth, max_depth = pred_depth_np.min(), pred_depth_np.max()
                pred_depth_np = ((pred_depth_np - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

                score_thresh = 0.6
                colored_mask = img_pred_vis.copy()
                for i in range(pred_boxes_np.shape[0]):
                    if pred_scores_np[i] > score_thresh:
                        box = pred_boxes_np[i]
                        label = pred_labels_np[i]
                        mask = pred_masks_np[i] > 0.5

                        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(img_pred_vis, (xmin, ymin), (xmax, ymax), class_to_colors[label - 1], 1)
                        colored_mask[mask] = np.random.randint(0, 255, 3)
                img_pred_vis = cv2.addWeighted(img_pred_vis, 0.8, colored_mask, 0.2, 0)

                filename_pred_box = os.path.join(output_dir, f"image_{image_id_val}_pred_box_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(filename_pred_box, img_pred_vis)
                filename_pred_depth = os.path.join(output_dir, f"image_{image_id_val}_pred_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(filename_pred_depth, pred_depth_np)
                cv2.imwrite(os.path.join(output_dir, "latest.png"), pred_depth_np)
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



