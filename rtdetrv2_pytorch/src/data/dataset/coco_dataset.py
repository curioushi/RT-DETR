"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os.path as osp

import torch
import torch.utils.data
import torch.nn.functional as F

import torchvision
import numpy as np
torchvision.disable_beta_transforms_warning()

from PIL import Image 
from faster_coco_eval.core import mask as coco_mask
from pycocotools import mask as pycoco_mask_api

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['CocoDetection']

def gaussian_kernel(height, width, center_y, center_x, sigma=1):
    diff_y = (np.arange(height).astype(np.float32) - center_y) ** 2
    diff_x = (np.arange(width).astype(np.float32) - center_x) ** 2
    diff_xy = diff_y[:, None] + diff_x[None, :]
    kernel = np.exp(-diff_xy / (2 * sigma ** 2))
    return kernel

def build_centernet_target(img, target):
    _, h, w = img.shape
    assert h % 8 == 0 and w % 8 == 0
    h2, w2 = h // 8, w // 8
    num_boxes = len(target["labels"])
    masks = np.zeros((num_boxes, h2, w2), dtype=np.float32)
    offsets = np.zeros((num_boxes, 2, h2, w2), dtype=np.float32)
    for i, coords2d in enumerate(target["coords2d"]):
        coords2d_full = coords2d.reshape(-1, 2).numpy()
        coords2d_full[:, 0] *= w
        coords2d_full[:, 1] *= h
        coords2d_down = coords2d_full / 8
        coords2d_down_int = coords2d_down.astype(int)
        coords2d_down_offset = coords2d_down - coords2d_down_int

        for ((x_idx, y_idx), (x_offset, y_offset)) in zip(coords2d_down_int, coords2d_down_offset):
            y_idx_min = max(0, y_idx - 2)
            y_idx_max = min(h2, y_idx + 3)
            x_idx_min = max(0, x_idx - 2)
            x_idx_max = min(w2, x_idx + 3)
            masks[i, y_idx_min:y_idx_max, x_idx_min:x_idx_max] = gaussian_kernel(
                                                                    y_idx_max - y_idx_min, 
                                                                    x_idx_max - x_idx_min, 
                                                                    y_idx - y_idx_min,
                                                                    x_idx - x_idx_min,
                                                                    1.0
                                                                )
            offsets[i, 0, y_idx_min:y_idx_max, x_idx_min:x_idx_max] = (x_offset - (np.array(range(x_idx_min, x_idx_max)) - x_idx))[None, :]
            offsets[i, 1, y_idx_min:y_idx_max, x_idx_min:x_idx_max] = (y_offset - (np.array(range(y_idx_min, y_idx_max)) - y_idx))[:, None]

    # img = img.numpy()
    # img = img.transpose(1, 2, 0)
    # img = (img * 255).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save(f"test_img.png")

    # for i, m in enumerate(mask):
    #     img = Image.fromarray(m * 255).convert("L")
    #     img.save(f"test_mask_{i}.png")
    
    # for i, o in enumerate(offset):
    #     o_uint16 = (o * 65535).astype(np.uint16)
    #     img = Image.fromarray(o_uint16[0])
    #     img.save(f"test_offset_{i}.png")

    masks = torch.from_numpy(masks)
    offsets = torch.from_numpy(offsets)

    target['masks'] = masks
    target['offsets'] = offsets



@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        build_centernet_target(img, target)
        return img, target

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        
        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s 

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def convert_coco_rle_to_masks(rle_segmentations_list, height, width):
    if not rle_segmentations_list:
        return torch.zeros((0, height, width), dtype=torch.uint8)

    masks_np = pycoco_mask_api.decode(rle_segmentations_list)
    if masks_np.ndim == 2:
        masks_np = masks_np[..., None]

    masks_torch = torch.as_tensor(masks_np, dtype=torch.uint8)
    return masks_torch.permute(2, 0, 1)


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
            
        labels = torch.tensor(labels, dtype=torch.int64)

        # Extract quads if present
        custom_coords2d = None
        if anno and "coords2d" in anno[0]: # Check if coords2d data exists
            custom_coords2d = [obj["coords2d"] for obj in anno]
            custom_coords2d = torch.as_tensor(custom_coords2d, dtype=torch.float32).reshape(-1, 4, 2)
            custom_coords2d[:, :, 0] /= w
            custom_coords2d[:, :, 1] /= h
            custom_coords2d = custom_coords2d.reshape(-1, 8)

        # Extract normals if present
        custom_centers = None
        custom_normals = None
        custom_x_axis = None
        custom_y_axis = None
        custom_offsets = None
        custom_sizes = None
        if anno and "pose" in anno[0] and "size" in anno[0]: # Check if normal data exists
            poses = np.array([np.array(obj["pose"]) for obj in anno])
            custom_centers = poses[:, :3, 3]
            custom_x_axis = poses[:, :3, 0]
            custom_y_axis = poses[:, :3, 1]
            custom_normals = poses[:, :3, 2]
            custom_offsets = -np.sum(custom_centers * custom_normals, axis=1)
            custom_sizes = np.array([obj["size"] for obj in anno])
            
            custom_centers = torch.as_tensor(custom_centers, dtype=torch.float32)
            custom_x_axis = torch.as_tensor(custom_x_axis, dtype=torch.float32)
            custom_y_axis = torch.as_tensor(custom_y_axis, dtype=torch.float32)
            custom_normals = torch.as_tensor(custom_normals, dtype=torch.float32)
            custom_offsets = torch.as_tensor(custom_offsets, dtype=torch.float32)
            custom_sizes = torch.as_tensor(custom_sizes, dtype=torch.float32)
        
        custom_camera_Ks = None
        if anno and "camera_K" in anno[0]:
            custom_camera_Ks = []
            for obj in anno:
                camera_K = obj["camera_K"]
                custom_camera_Ks.append([camera_K[0][0], camera_K[0][2], camera_K[1][1], camera_K[1][2]])
            custom_camera_Ks = torch.as_tensor(custom_camera_Ks, dtype=torch.float32).reshape(-1, 4)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_rle_to_masks(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if custom_coords2d is not None:
            custom_coords2d = custom_coords2d[keep]
        if custom_normals is not None:
            custom_normals = custom_normals[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        if custom_coords2d is not None:
            target["coords2d"] = custom_coords2d
        if custom_centers is not None:
            target["centers"] = custom_centers
        if custom_normals is not None:
            target["normals"] = custom_normals
        if custom_x_axis is not None:
            target["x_axis"] = custom_x_axis
        if custom_y_axis is not None:
            target["y_axis"] = custom_y_axis
        if custom_offsets is not None:
            target["offsets"] = custom_offsets
        if custom_sizes is not None:
            target["sizes"] = custom_sizes
        if custom_camera_Ks is not None:
            target["camera_Ks"] = custom_camera_Ks

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])
        # assert target['boxes'].shape[0] == target['coords'].shape[0]
    
        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
