"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from typing import List

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .hybrid_encoder import ConvNormLayer, CSPRepLayer

from ...core import register

__all__ = ['RTDETRTransformerv2']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PointNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=(32, 64, 128), featmap_dim=256):
        super(PointNet, self).__init__()
        assert len(hidden_dim) == 3, "hidden_dim must be a tuple of length 3"
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim[0], kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim[0], hidden_dim[1], kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim[1], hidden_dim[2], kernel_size=1),
            nn.GELU(),
        )
        self.featmap_head = nn.Sequential(
            nn.Conv1d(featmap_dim, hidden_dim[2], kernel_size=1),
            nn.GELU(),
        )
        self.fuse_head = nn.Sequential(
            nn.Conv1d(2 * hidden_dim[2], 2 * hidden_dim[2], kernel_size=1),
            nn.GELU(),
        )
        self.center_offset_head = nn.Linear(2 * hidden_dim[2], 3)
        self.rotation_head = nn.Linear(2 * hidden_dim[2], 6)
        self.size_head = nn.Linear(2 * hidden_dim[2], 2)

        nn.init.constant_(self.center_offset_head.weight, 0)
        nn.init.constant_(self.rotation_head.weight, 0)
        nn.init.constant_(self.size_head.weight, 0)
        nn.init.constant_(self.center_offset_head.bias, 0)
        nn.init.constant_(self.rotation_head.bias, 0)
        nn.init.constant_(self.size_head.bias, 0)

    def forward(self, query_points, query_masks, featmap):
        """
        query_points: B x NQ x 3 x NP
        query_masks: B x NQ x NP
        featmap: B x NQ x featmap_dim
        """
        B, NQ, _, NP = query_points.shape
        featmap_dim = featmap.shape[2]
        query_points = query_points.reshape(-1, 3, NP)
        query_masks = query_masks.reshape(-1, NP)
        featmap = self.featmap_head(featmap.reshape(-1, featmap_dim, 1)).expand(-1, -1, NP) # B*NQ, C, NP
        x = self.mlp(query_points) # B*NQ, C, NP
        x = torch.cat([x, featmap], dim=1) # B*NQ, 2*C, NP
        x = self.fuse_head(x)
        x = x * query_masks.unsqueeze(1)
        x = F.max_pool1d(x, int(x.size(2))).squeeze(-1) # B*NQ, 2*C
        center_offsets = self.center_offset_head(x) # B*NQ, 3
        rotations = self.rotation_head(x) # B*NQ, 6
        sizes = self.size_head(x) # B*NQ, 2
        center_offsets = center_offsets.reshape(B, NQ, 3)
        rotations = rotations.reshape(B, NQ, 6)
        sizes = sizes.reshape(B, NQ, 2)
        return center_offsets, rotations, sizes


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                mask_embedding,
                bbox_head,
                score_head,
                quad_head,
                weights_head,
                mask_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_quads = []
        dec_out_weights = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            if i == 0:
                cx, cy, w, h = inter_ref_bbox[:, :, 0], inter_ref_bbox[:, :, 1], inter_ref_bbox[:, :, 2], inter_ref_bbox[:, :, 3]
                ref_quads_detach = torch.stack([cx + w/2, cy + h/2,
                                                cx - w/2, cy + h/2,
                                                cx - w/2, cy - h/2,
                                                cx + w/2, cy - h/2], dim=-1)
            inter_ref_quads = quad_head[i](output) + ref_quads_detach
            
            predicted_weights = weights_head[i](output)

            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out_weights.append(predicted_weights)

                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    dec_out_quads.append(inter_ref_quads)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
                    dec_out_quads.append(quad_head[i](output) + ref_quads)

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_quads.append(inter_ref_quads)
                dec_out_weights.append(predicted_weights)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
            ref_quads = inter_ref_quads
            ref_quads_detach = inter_ref_quads.detach()
        
        mask_query = mask_head(output)
        dec_out_masks = torch.einsum('bqc,bchw->bqhw', mask_query, mask_embedding)

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(dec_out_quads), torch.stack(dec_out_weights), dec_out_masks


@register()
class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True, 
                 cross_attn_method='default', 
                 query_select_method='default'):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # FPN layers
        depth_mult = 1.0 # default value from HybridEncoder
        expansion = 0.5 # default value from HybridEncoder
        act_fn = 'silu' # default value from HybridEncoder

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act_fn))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act_fn, expansion=expansion)
            )
        self.high_res_lateral_conv1 = ConvNormLayer(64, hidden_dim, 1, 1, act=act_fn)
        self.high_res_lateral_conv2 = ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act_fn)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])
        self.dec_quad_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 8, 3) for _ in range(num_layers)
        ])
        self.dec_weights_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])
        self.dec_mask_head = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.dec_mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.dec_depth_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.dec_pointnet = PointNet(input_dim=3, hidden_dim=(32, 64, 128), featmap_dim=hidden_dim)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()
        
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        
        for _quad in self.dec_quad_head:
            init.constant_(_quad.layers[-1].weight, 0)
            init.constant_(_quad.layers[-1].bias, 0)
        
        for _reg_norm in self.dec_weights_head:
            init.constant_(_reg_norm.layers[-1].weight, 0)
            init.constant_(_reg_norm.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return proj_feats, feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
            
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:            
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords
    
    def _depth_to_xyz(self, depth: torch.Tensor, targets: List[dict]):
        camera_Ks = torch.stack([t['camera_Ks'][0] for t in targets])
        orig_sizes = torch.stack([t['orig_size'] for t in targets])
        fx, cx, fy, cy = camera_Ks[:, 0], camera_Ks[:, 1], camera_Ks[:, 2], camera_Ks[:, 3]
        ws, hs = orig_sizes[:, 0], orig_sizes[:, 1]
        fx = fx / ws
        cx = cx / ws
        fy = fy / hs
        cy = cy / hs

        b, h, w = depth.shape
        device = depth.device

        # Create v and u coordinate maps
        v_coords = (torch.arange(h, dtype=depth.dtype, device=device) / h).unsqueeze(1).expand(h, w)
        u_coords = (torch.arange(w, dtype=depth.dtype, device=device) / w).unsqueeze(0).expand(h, w)

        # Reshape camera parameters for broadcasting
        # fx, cx, fy, cy have shape (b), reshape to (b, 1, 1)
        fx_exp = fx.view(b, 1, 1)
        cx_exp = cx.view(b, 1, 1)
        fy_exp = fy.view(b, 1, 1)
        cy_exp = cy.view(b, 1, 1)

        # Expand u_coords and v_coords to (b, h, w) for broadcasting
        u_coords_exp = u_coords.unsqueeze(0).expand(b, h, w)
        v_coords_exp = v_coords.unsqueeze(0).expand(b, h, w)
        
        # Calculate normalized coordinates
        x_normalized = (u_coords_exp - cx_exp) / fx_exp
        y_normalized = (v_coords_exp - cy_exp) / fy_exp

        # Calculate 3D coordinates in camera frame
        depth = torch.exp(depth) - 1
        x_cam = x_normalized * depth
        y_cam = y_normalized * depth
        z_cam = depth # Depth is the z-coordinate

        # Stack to form xyz_mapping (b, 3, h, w)
        xyz_mapping = torch.stack([x_cam, y_cam, z_cam], dim=1)
        
        return xyz_mapping

    def forward(self, feats, feat_high_res, targets=None):
        # input projection and embedding
        proj_feats, memory, spatial_shapes = self._get_encoder_input(feats)
        
        # FPN operation on proj_feats
        fpn_inner_outs = [proj_feats[-1]]
        for idx in range(self.num_levels - 1, 0, -1):
            feat_high = fpn_inner_outs[0]
            feat_low = proj_feats[idx - 1]
            conv_idx = (self.num_levels - 1) - idx
            feat_high = self.lateral_convs[conv_idx](feat_high)
            fpn_inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[conv_idx](torch.concat([upsample_feat, feat_low], dim=1))
            fpn_inner_outs.insert(0, inner_out)
        out_featmap = fpn_inner_outs[0]
        feat_high_res1 = self.high_res_lateral_conv1(feat_high_res)
        feat_high_res2 = F.interpolate(self.high_res_lateral_conv2(out_featmap), scale_factor=2., mode='nearest')
        out_featmap = feat_high_res1 + feat_high_res2

        mask_embedding = self.dec_mask_embed(out_featmap.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out_depth = self.dec_depth_head(out_featmap.permute(0, 2, 3, 1)).squeeze(-1)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits, out_quads, out_weights, out_masks = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            mask_embedding,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_quad_head,
            self.dec_weights_head,
            self.dec_mask_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_quads, out_quads = torch.split(out_quads, dn_meta['dn_num_split'], dim=2)
            dn_out_weights, out_weights = torch.split(out_weights, dn_meta['dn_num_split'], dim=2)
            dn_out_masks, out_masks = torch.split(out_masks, dn_meta['dn_num_split'], dim=1)
        
        topk = 1000
        nq = out_masks.shape[1]
        h, w = out_featmap.shape[2:]
        out_depth_detach = out_depth.detach()
        out_masks_detach = out_masks.detach()
        xyz_points = self._depth_to_xyz(out_depth_detach, targets).flatten(2) # b, 3, h*w
        query_weights = F.sigmoid(out_masks_detach).flatten(2)  # b, nq, h*w
        rgb_embeddings = torch.einsum('bqn,bcn->bqc', query_weights, out_featmap.flatten(2)) / (query_weights.sum(dim=-1, keepdim=True) + 1e-1)
        _, top_k_indices = torch.topk(query_weights + torch.rand_like(query_weights) * 0.2, k=topk, dim=-1) # b, nq, topk
        query_weights = torch.gather(query_weights, dim=-1, index=top_k_indices) # b, nq, topk
        xyz_points = torch.gather(xyz_points.unsqueeze(1).expand(-1, nq, -1, -1), dim=-1, 
                                  index=top_k_indices.unsqueeze(2).expand(-1, -1, 3, -1)) # b, nq, 3, topk
        query_weights_sum = query_weights.sum(dim=-1) # b, nq
        query_centers = (xyz_points * query_weights.unsqueeze(2)).sum(dim=-1) / query_weights_sum.unsqueeze(-1) # b, nq, 3
        query_centered_points = xyz_points - query_centers.unsqueeze(-1) # b, nq, 3, topk
        center_offset, rotation, size = self.dec_pointnet(query_centered_points, query_weights, rgb_embeddings)
        query_centers = query_centers + center_offset
        query_rotations = rotation
        query_sizes = size
        
        out = {'pred_logits': out_logits[-1], 
               'pred_boxes': out_bboxes[-1], 
               'pred_quads': out_quads[-1], 
               'pred_weights': out_weights[-1],
               'pred_depths': out_depth,
               'pred_masks': out_masks,
               'pred_centers': query_centers,
               'pred_rotations': query_rotations,
               'pred_sizes': query_sizes,
               }

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_quads[:-1], out_weights[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                # out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_out_normals)
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_quad=None, outputs_weight=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_quad is not None and outputs_weight is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_quads': c, 'pred_weights': d}
                    for a, b, c, d in zip(outputs_class, outputs_coord, outputs_quad, outputs_weight)]
        elif outputs_quad is None and outputs_weight is None:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class, outputs_coord)]
        else:
            raise ValueError('outputs_quad and outputs_weight must be either both None or both not None')
