"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import cv2
import shutil
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np 
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def fix_points_order(points: list[tuple[int, int]], label: int) -> list[tuple[int, int]]:
    """Fix points order based on label"""
    order = None
    if label in [0, 1, 2]:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        sorted_idx_x = np.argsort(xs)
        if ys[sorted_idx_x[0]] < ys[sorted_idx_x[1]]:
            left_top_idx = sorted_idx_x[0]
            left_bottom_idx = sorted_idx_x[1]
        else:
            left_top_idx = sorted_idx_x[1]
            left_bottom_idx = sorted_idx_x[0]
        if ys[sorted_idx_x[2]] < ys[sorted_idx_x[3]]:
            right_top_idx = sorted_idx_x[2]
            right_bottom_idx = sorted_idx_x[3]
        else:
            right_top_idx = sorted_idx_x[3]
            right_bottom_idx = sorted_idx_x[2]
        order = [left_top_idx, left_bottom_idx, right_bottom_idx, right_top_idx]
    elif label == 3:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        sorted_idx_x = np.argsort(xs)
        sorted_idx_y = np.argsort(ys)
        if xs[sorted_idx_y[0]] < xs[sorted_idx_y[3]]:
            p0_idx = sorted_idx_y[0]
            p2_idx = sorted_idx_y[3]
            if xs[sorted_idx_y[1]] < xs[sorted_idx_y[2]]:
                p1_idx = sorted_idx_y[1]
                p3_idx = sorted_idx_y[2]
            else:
                p1_idx = sorted_idx_y[2]
                p3_idx = sorted_idx_y[1]
            order = [p0_idx, p1_idx, p2_idx, p3_idx]
        else:
            p1_idx = sorted_idx_y[3]
            p3_idx = sorted_idx_y[0]
            if xs[sorted_idx_y[1]] < xs[sorted_idx_y[2]]:
                p0_idx = sorted_idx_y[1]
                p2_idx = sorted_idx_y[2]
            else:
                p0_idx = sorted_idx_y[2]
                p2_idx = sorted_idx_y[1]
            order = [p0_idx, p1_idx, p2_idx, p3_idx]
    else:
        raise ValueError(f"Invalid label: {label}")

    points = [points[i] for i in order]
    return points

def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes, targets=None):
            outputs = self.model(images, targets)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    im_files = sorted(glob(os.path.join(args.im_dir, '*.jpg')))
    im_files += sorted(glob(os.path.join(args.im_dir, '*.png')))
    for im_file in tqdm(im_files):
        im_pil = Image.open(im_file).convert('RGB')

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        sample = im_data

        with torch.no_grad():
            output = model(sample, orig_size)

        labels, boxes, scores, quads, masks, offsets = output
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        quads = quads[0]
        masks = masks[0]
        offsets = offsets[0]

        masks = F.sigmoid(masks)
        masks_max_pool = F.max_pool2d(masks.unsqueeze(1), kernel_size=5, padding=2, stride=1).squeeze(1)
        masks_peak = masks * (masks == masks_max_pool).float()

        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        quads = quads.cpu().numpy()
        masks = masks.cpu().numpy()
        masks_peak = masks_peak.cpu().numpy()
        offsets = offsets.cpu().numpy()

        class_to_colors = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 255, 0),
        }

        viz_img = np.array(im_pil)

        top3_indices = np.argsort(scores)[-3:]
        for i in top3_indices:
            label = labels[i]
            # Find the four points with maximum values in pred_masks_peak_np
            mask_flat = masks_peak[i].flatten()
            top_4_indices = np.argsort(mask_flat)[-4:]
            ys, xs = np.unravel_index(top_4_indices, masks_peak[i].shape)
            if len(xs) == 4:
                x_offsets = offsets[i, 0, ys, xs]
                y_offsets = offsets[i, 1, ys, xs]
                quads = np.stack([xs + x_offsets, ys + y_offsets], axis=1) * 8
                quads[:, 0] *= w / 640
                quads[:, 1] *= h / 640
                quads = np.array(fix_points_order(quads.tolist(), label))
                cv2.polylines(viz_img, [quads.astype(np.int32)], True, class_to_colors[label], 2)
            

        cv2.imwrite(f"output/predict/{os.path.basename(im_file)}", viz_img)


        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-i', '--im-dir', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()

    shutil.rmtree("output/predict", ignore_errors=True)
    os.makedirs("output/predict")
    main(args)
