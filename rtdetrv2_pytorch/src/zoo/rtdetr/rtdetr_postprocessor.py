"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes, quads, masks, offsets = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_quads'], outputs['pred_masks'], outputs['pred_offsets']
        bs, nq, h, w = masks.shape
        masks = masks.flatten(2)
        offsets = offsets.flatten(2)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            quads = quads.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, quads.shape[-1]))
            masks = masks.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, masks.shape[-1]))
            offsets = offsets.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, offsets.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                quads = torch.gather(quads, dim=1, index=index.unsqueeze(-1).tile(1, 1, quads.shape[-1]))
                masks = torch.gather(masks, dim=1, index=index.unsqueeze(-1).tile(1, 1, masks.shape[-1]))
                offsets = torch.gather(offsets, dim=1, index=index.unsqueeze(-1).tile(1, 1, offsets.shape[-1]))

        masks = masks.reshape(bs, nq, h, w)
        offsets = offsets.reshape(bs, nq, 2, h, w)
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores, quads, masks, offsets

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco, quad, mask, offset in zip(labels, boxes, scores, quads, masks, offsets):
            result = dict(labels=lab, boxes=box, scores=sco, quads=quad, masks=mask, offsets=offset)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
