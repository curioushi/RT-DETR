"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import json
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np 
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
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
            
        def forward(self, images, orig_target_sizes, targets):
            outputs = self.model(images, targets)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    im_files = sorted(glob(os.path.join(args.im_dir, '*.png')))
    xyz_files = sorted(glob(os.path.join(args.im_dir, '../xyz', '*.npz')))
    # prediction = dict()
    for im_file, xyz_file in tqdm(zip(im_files, xyz_files)):
        im_pil = Image.open(im_file).convert('RGB')
        # depth preprocessing
        depth = np.load(xyz_file)['xyz_mapping'][:, :, 2]
        depth = torch.from_numpy(depth).float().unsqueeze(0) # (1, 1024, 1024)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(640, 640),
            mode='nearest'
        ).squeeze(0)
        depth = torch.log(torch.clamp(depth, 0.0, 100.0) + 1)
        # sample 2% of the depth points
        sparse_depth = torch.zeros_like(depth)
        num_points = depth.numel()
        num_samples = int(num_points * 0.02)
        if num_samples > 0:
            row_indices = torch.randint(0, depth.shape[1], (num_samples,))
            col_indices = torch.randint(0, depth.shape[2], (num_samples,))
            sparse_depth[:, row_indices, col_indices] = depth[:, row_indices, col_indices]
            

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        sparse_depth = sparse_depth.unsqueeze(0).to(args.device)

        sample = torch.cat([im_data, sparse_depth], dim=1)
        # hack
        targets = [{
            "orig_size": torch.tensor([1024, 1024]).float().to(args.device),
            "camera_Ks": torch.tensor([295.6033378250885, 512.0, 295.6033378250885, 512.0]).unsqueeze(0).to(args.device),
        }]

        with torch.no_grad():
            output = model(sample, orig_size, targets)
        labels, boxes, scores, quads, weights, depths, masks, centers, rotations, sizes = output
        boxes = boxes.squeeze(0).cpu().numpy()
        labels = labels.squeeze(0).cpu().numpy()
        scores = scores.squeeze(0).cpu().numpy()
        centers = centers.squeeze(0).cpu().numpy()
        rotations = rotations.squeeze(0).cpu().numpy()
        sizes = sizes.squeeze(0).cpu().numpy()
        masks = masks.squeeze(0).cpu().numpy()

        indices = scores > 0.6
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        centers = centers[indices]
        rotations = rotations[indices]
        sizes = sizes[indices]
        masks = masks[indices]
        depth = (torch.exp(depths) - 1).squeeze(0).cpu().numpy()
        np.savez_compressed(f'prediction_{os.path.basename(im_file).replace("png", "npz")}', 
                            boxes=boxes,
                            labels=labels,
                            scores=scores,
                            centers=centers,
                            rotations=rotations,
                            sizes=sizes,
                            depth=depth, 
                            masks=masks)

        # draw([im_pil], labels, boxes, scores)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-i', '--im-dir', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
