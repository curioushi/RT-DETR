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
        labels, boxes, scores, quads = output
        boxes = boxes.squeeze(0).cpu().numpy()
        labels = labels.squeeze(0).cpu().numpy()
        scores = scores.squeeze(0).cpu().numpy()
        quads = quads.squeeze(0).cpu().numpy()

        indices = scores > 0.6
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        quads = quads[indices]
        quads = quads.reshape(-1, 4, 2)
        quads[:, :, 0] *= w
        quads[:, :, 1] *= h

        class_to_colors = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 255, 0),
        }

        viz_img = np.array(im_pil)
        for label, quad, in zip(labels, quads):
            cv2.polylines(viz_img, [quad.astype(np.int32)], True, class_to_colors[label], 2)

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
