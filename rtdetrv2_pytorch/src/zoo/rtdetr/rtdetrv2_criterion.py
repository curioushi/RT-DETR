"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_quads(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_quads' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_quads = outputs['pred_quads'][idx]
        src_weights = outputs['pred_weights'][idx]

        target_quads = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_weights_list = []
        for t, (_, i) in zip(targets, indices):
            if 'weights' not in t:
                raise ValueError("Target weights not found in one or more targets.")
            if i.numel() > 0: # only gather if there are matched indices for this target
                target_weights_list.append(t['weights'][i])

        losses = {}
        loss_quads = F.l1_loss(src_quads, target_quads, reduction='none')

        if not target_weights_list:
            loss_weights = torch.zeros(1, device=src_weights.device, requires_grad=True)[0]
            loss_length_consistency = torch.zeros(1, device=src_weights.device, requires_grad=True)[0]
        else:
            target_weights = torch.cat(target_weights_list, dim=0)
            assert src_weights.shape[0] == target_weights.shape[0], 'src_weights and target_weights shape mismatch'

            loss_weights = F.l1_loss(src_weights, target_weights, reduction='none')
            loss_length_consistency = self.loss_consistency_per_quad(outputs, targets, indices)
            mask_good_quads = (loss_weights.mean(axis=-1) < 0.1) & (loss_quads.max(axis=-1)[0] < 0.01)
            loss_length_consistency = (loss_length_consistency * mask_good_quads).sum() / (mask_good_quads.sum() + 1e-7)

        losses['loss_quads'] = loss_quads.sum() / num_boxes
        losses['loss_weights'] = loss_weights.sum() / num_boxes
        losses['loss_length_consistency'] = loss_length_consistency

        return losses
    
    def loss_consistency_per_quad(self, outputs, targets, indices):
        assert 'pred_quads' in outputs, "'pred_quads' not found in outputs."
        assert 'pred_weights' in outputs, "'pred_weights' not found in outputs."
        
        idx = self._get_src_permutation_idx(indices)
        src_quads_flat = outputs['pred_quads'][idx] # Shape: (M, 8)
        src_weights = outputs['pred_weights'][idx]    # Shape: (M, 4), for (a, b, c, d)

        if src_quads_flat.shape[0] == 0:
            # No matched predictions, so loss is 0
            return {'loss_quads_3d': torch.tensor(0.0, device=src_quads_flat.device, dtype=src_quads_flat.dtype)}

        # Collect camera K and original sizes for matched predictions
        # This logic mirrors the structure seen in the user-provided context for these variables
        target_camera_Ks_for_cat = []
        target_orig_sizes_for_cat = []
        for t, (_, matched_target_indices_in_image) in zip(targets, indices):
            # matched_target_indices_in_image are indices into this specific target `t`'s annotations
            if matched_target_indices_in_image.numel() > 0:
                if 'camera_Ks' not in t:
                    raise ValueError(f"Target 'camera_Ks' not found in target dict (keys: {list(t.keys())}) for an image with matches.")
                if 'orig_size' not in t:
                    raise ValueError(f"Target 'orig_size' not found in target dict (keys: {list(t.keys())}) for an image with matches.")
                
                target_camera_Ks_for_cat.append(t['camera_Ks'][matched_target_indices_in_image])
                num_matches_in_image = len(matched_target_indices_in_image)
                target_orig_sizes_for_cat.append(t['orig_size'].unsqueeze(0).repeat(num_matches_in_image, 1))
        
        # If src_quads_flat.shape[0] > 0, then target_camera_Ks_for_cat and target_orig_sizes_for_cat must be non-empty,
        # otherwise an error would have been raised or src_quads_flat would have been empty.
        norm_target_camera_Ks_unnormalized = torch.cat(target_camera_Ks_for_cat, dim=0)
        target_orig_sizes_batched = torch.cat(target_orig_sizes_for_cat, dim=0)
        
        # Normalize camera intrinsics using the formula from the provided context
        # Adding a small epsilon for stability during division
        denominator_K_norm = target_orig_sizes_batched.repeat(1, 2)
        norm_target_camera_Ks = norm_target_camera_Ks_unnormalized / denominator_K_norm
        # norm_target_camera_Ks is expected to be [fx, cx, fy, cy] (all normalized)

        fx = norm_target_camera_Ks[:, 0:1] # Shape: (M, 1)
        cx = norm_target_camera_Ks[:, 1:2] # Shape: (M, 1)
        fy = norm_target_camera_Ks[:, 2:3] # Shape: (M, 1)
        cy = norm_target_camera_Ks[:, 3:4] # Shape: (M, 1)
        
        src_quads_reshaped = src_quads_flat.reshape(-1, 4, 2) # Shape: (M, 4, 2)
        u = src_quads_reshaped[..., 0] # Shape: (M, 4) - normalized image coordinates
        v = src_quads_reshaped[..., 1] # Shape: (M, 4) - normalized image coordinates

        Dx = (u - cx) / fx # Shape: (M, 4)
        Dy = (v - cy) / fy # Shape: (M, 4)
        Dz = torch.ones_like(Dx)    # Shape: (M, 4) - using Dx to get shape, device, dtype
        rays_d = torch.stack([Dx, Dy, Dz], dim=-1) # Shape: (M, 4, 3)

        n_plane = src_weights[:, :3]       # Shape: (M, 3) - normal vector (a,b,c)
        d_plane = src_weights[:, 3:4]      # Shape: (M, 1) - offset d

        # Denominator for t: n_plane . D = a*Dx + b*Dy + c*Dz
        # n_plane (M,3), rays_d (M,4,3). Einsum is efficient for this.
        n_plane_dot_D = torch.einsum('mi,mji->mj', n_plane, rays_d) # Shape: (M, 4)

        # Parameter t for ray equation P(t) = O + tD (O is origin). So, t = -d_plane / (n_plane . D)
        # Stabilize division by zero or very small denominators
        abs_denom = torch.abs(n_plane_dot_D)
        signed_epsilon = torch.copysign(torch.full_like(n_plane_dot_D, 1e-7), n_plane_dot_D)
        safe_denominator = torch.where(abs_denom < 1e-7, signed_epsilon, n_plane_dot_D)
        
        t = -d_plane / safe_denominator # Shape: (M, 4)
        
        P_intersect = t.unsqueeze(-1) * rays_d # Shape: (M, 4, 3) - These are p1, p2, p3, p4 for each quad

        edges_vec = P_intersect - torch.roll(P_intersect, shifts=-1, dims=1) # Shape: (M, 4, 3)
        edge_lengths = torch.norm(edges_vec, p=2, dim=-1) # Shape: (M, 4)
        
        l1 = edge_lengths[:, 0] 
        l2 = edge_lengths[:, 1] 
        l3 = edge_lengths[:, 2] 
        l4 = edge_lengths[:, 3] 

        loss_length_per_quad = (torch.abs(l1 - l3) / (l1 + l3 + 1e-7)) + \
                               (torch.abs(l2 - l4) / (l2 + l4 + 1e-7))

        # inner1 = torch.einsum('mi,mi->m', edges_vec[:, 0], edges_vec[:, 1]) / (l1 * l2 + 1e-7)
        # inner2 = torch.einsum('mi,mi->m', edges_vec[:, 1], edges_vec[:, 2]) / (l2 * l3 + 1e-7)
        # inner3 = torch.einsum('mi,mi->m', edges_vec[:, 2], edges_vec[:, 3]) / (l3 * l4 + 1e-7)
        # inner4 = torch.einsum('mi,mi->m', edges_vec[:, 3], edges_vec[:, 0]) / (l4 * l1 + 1e-7)
        # loss_angle_per_quad = (torch.abs(inner1) + torch.abs(inner2) + torch.abs(inner3) + torch.abs(inner4)) / 4
        
        
        return loss_length_per_quad
    
    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_masks' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_masks = outputs['pred_masks'][idx]
        h, w = src_masks.shape[1:]
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_masks = F.interpolate(target_masks.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
        loss_mask_bce = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='none').flatten(1)
        return {'loss_mask_bce': loss_mask_bce.mean(axis=1).sum() / num_boxes}
    
    def loss_depth(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_depths' in outputs
        if len(targets) > 0: 
            src_depth = outputs['pred_depths']
            target_depth = torch.cat([t['depth'] for t in targets], dim=0)
            valid_depth_mask = torch.cat([t['valid_depth_mask'] for t in targets], dim=0).float()
            bs, h, w = src_depth.shape
            target_depth = F.interpolate(target_depth.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
            valid_depth_mask = F.interpolate(valid_depth_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
            loss_depth = F.l1_loss(src_depth, target_depth, reduction='none') * valid_depth_mask
            return {'loss_depth': loss_depth.sum() / (valid_depth_mask.sum() + 1e-7)}
        else:
            return {}
    
    def loss_quads3d(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_centers' in outputs, "'pred_centers' not found in outputs."
        assert 'pred_covariances' in outputs, "'pred_covariances' not found in outputs."
        
        idx = self._get_src_permutation_idx(indices)
        src_centers = outputs['pred_centers'][idx]
        src_covariances = outputs['pred_covariances'][idx]

        target_centers = torch.cat([t['centers'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_covariances = torch.cat([t['covariances'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_covariances = target_covariances.flatten(1)[:, [0, 1, 2, 4, 5, 8]]

        loss_quads3d_center = F.l1_loss(src_centers, target_centers, reduction='none')
        loss_quads3d_covariance = F.l1_loss(src_covariances, target_covariances, reduction='none')
        return {'loss_quads3d_center': loss_quads3d_center.sum() / num_boxes,
                'loss_quads3d_covariance': loss_quads3d_covariance.sum() / num_boxes}
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'quads': self.loss_quads,
            'masks': self.loss_masks,
            'depth': self.loss_depth,
            'quads3d': self.loss_quads3d,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    if loss in ['masks', 'depth', 'quads3d']:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss in ['quads', 'masks', 'depth', 'quads3d']:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    if loss in ['quads', 'masks', 'depth', 'quads3d']:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
