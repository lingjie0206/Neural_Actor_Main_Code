# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
logger = logging.getLogger(__name__)

import cv2, math, time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.utils import with_torch_seed
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.fairnr_model import BaseModel


@register_model('nerf')
class NeRFModel(BaseModel):
    """ This is a simple re-implementation of the vanilla NeRF
    """
    ENCODER = 'volume_encoder'
    READER = 'image_reader'
    FIELD = 'radiance_field'
    RAYMARCHER = 'volume_rendering'

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--fixed-num-samples', type=int, 
            help='number of samples for the first pass along the ray.')
        parser.add_argument('--fixed-fine-num-samples', type=int,
            help='sample a fixed number of points for each ray in hierarchical sampling, e.g. 64, 128.')
        parser.add_argument('--reduce-fine-for-missed', action='store_true',
            help='if set, the number of fine samples is discounted based on foreground probability only.')

    def preprocessing(self, **kwargs):
        return self.encoder.precompute(**kwargs)

    def intersecting(self, ray_start, ray_dir, encoder_states, **kwargs):
        S = ray_dir.size(0)
        ray_start, ray_dir, intersection_outputs, hits = \
            self.encoder.ray_intersect(ray_start, ray_dir, encoder_states)
        
        if self.reader.no_sampling and self.training:  # sample points after ray-voxel intersection
            uv, size = kwargs['uv'], kwargs['size']
            mask = hits.reshape(*uv.size()[:2], uv.size(-1))

            # sample rays based on voxel intersections
            sampled_uv, sampled_masks = self.reader.sample_pixels(
                uv, size, mask=mask, return_mask=True)
            sampled_masks = sampled_masks.reshape(uv.size(0), -1).bool()
            hits, sampled_masks = hits[sampled_masks].reshape(S, -1), sampled_masks.unsqueeze(-1)
            intersection_outputs = {name: outs[sampled_masks.expand_as(outs)].reshape(S, -1, outs.size(-1)) 
                if outs.dim() == 3 else outs[sampled_masks.squeeze(-1)].reshape(S, -1)
                for name, outs in intersection_outputs.items()}
            ray_start = ray_start[sampled_masks.expand_as(ray_start)].reshape(S, -1, 3)
            ray_dir = ray_dir[sampled_masks.expand_as(ray_dir)].reshape(S, -1, 3)
        
        else:
            sampled_uv = None
        
        return ray_start, ray_dir, intersection_outputs, hits, sampled_uv

    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        # sample points and use middle point approximation
        with with_torch_seed(self.unique_seed):  # make sure each GPU sample differently.
            samples = self.encoder.ray_sample(intersection_outputs)
        field = self.field_fine if fine and (self.field_fine is not None) else self.field 
        all_results = self.raymarcher(
            self.encoder, field, ray_start, ray_dir, samples, encoder_states
        )
        return samples, all_results

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        # this function is basically the same as that in NSVF model.
        depth = samples.get('original_point_depth', samples['sampled_point_depth'])
        dists = samples.get('original_point_distance', samples['sampled_point_distance'])
        intersection_outputs['min_depth'] = depth - dists * .5
        intersection_outputs['max_depth'] = depth + dists * .5
        intersection_outputs['intersected_voxel_idx'] = samples['sampled_point_voxel_idx'].contiguous()
        safe_probs = all_results['probs'] + 1e-8  # HACK: make a non-zero distribution
        # safe_probs = all_results['probs'] + 1e-5  # NeRF used 1e-5, will this make a change?
        intersection_outputs['probs'] = safe_probs / safe_probs.sum(-1, keepdim=True)
        intersection_outputs['steps'] = safe_probs.new_ones(*safe_probs.size()[:-1]) 
        if getattr(self.args, "fixed_fine_num_samples", 0) > 0:
            intersection_outputs['steps'] = intersection_outputs['steps'] * self.args.fixed_fine_num_samples
        if getattr(self.args, "reduce_fine_for_missed", False):
            intersection_outputs['steps'] = intersection_outputs['steps'] * safe_probs.sum(-1)
        return intersection_outputs

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # vanilla nerf hits everything. so no need to fill_in
        S, V, P = sizes
        fullsize = S * V * P
        
        all_results['missed'] = all_results['missed'].view(S, V, P)
        all_results['colors'] = all_results['colors'].view(S, V, P, 3)
        all_results['depths'] = all_results['depths'].view(S, V, P)
        if 'z' in all_results:
            all_results['z'] = all_results['z'].view(S, V, P)
        BG_DEPTH = self.field.bg_color.depth
        bg_color = self.field.bg_color(all_results['colors'])
        all_results['colors'] += all_results['missed'].unsqueeze(-1) * bg_color.reshape(fullsize, 3).view(S, V, P, 3)
        all_results['depths'] += all_results['missed'] * BG_DEPTH
        if 'normal' in all_results:
            all_results['normal'] = all_results['normal'].view(S, V, P, 3)
        return all_results

    def add_other_logs(self, all_results):
        return {}


@register_model_architecture("nerf", "nerf_base")
def base_architecture(args):
    # parameter needs to be changed
    args.near = getattr(args, "near", 2)
    args.far = getattr(args, "far", 4)
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 64)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 128)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", True)
    args.use_fine_model = getattr(args, "use_fine_model", True)

    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
    args.density_embed_dim = getattr(args, "density_embed_dim", 128)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)

    # API Update: fix the number of layers
    args.feature_layers = getattr(args, "feature_layers", 1)
    args.texture_layers = getattr(args, "texture_layers", 3)
    
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.background_depth = getattr(args, "background_depth", 5.0)
    
    # raymarcher
    args.discrete_regularization = getattr(args, "discrete_regularization", False)
    args.deterministic_step = getattr(args, "deterministic_step", False)
    args.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0)

    # reader
    args.pixel_per_view = getattr(args, "pixel_per_view", 2048)
    args.sampling_on_mask = getattr(args, "sampling_on_mask", 0.0)
    args.sampling_at_center = getattr(args, "sampling_at_center", 1.0)
    args.sampling_on_bbox = getattr(args, "sampling_on_bbox", False)
    args.sampling_patch_size = getattr(args, "sampling_patch_size", 1)
    args.sampling_skipping_size = getattr(args, "sampling_skipping_size", 1)

    # others
    args.chunk_size = getattr(args, "chunk_size", 64)
    args.valid_chunk_size = getattr(args, "valid_chunk_size", 64)

@register_model_architecture("nerf", "nerf_deep")
def nerf_deep_architecture(args):
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    base_architecture(args)
    
@register_model_architecture("nerf", "nerf_nerf")
def nerf_nerf_architecture(args):
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.texture_layers = getattr(args, "texture_layers", 0)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    base_architecture(args)

@register_model_architecture("nerf", "nerf_xyzn_nope")
def nerf2_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:0:3, normal:0:3, sigma:0:1, ray:4")
    base_architecture(args)


@register_model('sdf_nerf')
class SDFNeRFModel(NeRFModel):

    FIELD = "sdf_radiance_field"


@register_model_architecture("sdf_nerf", "sdf_nerf")
def sdf_nsvf_architecture(args):
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    nerf2_architecture(args)

@register_model_architecture("sdf_nerf", "sdf_nerf_coarse")
def sdf_nsvf2_architecture(args):
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 256)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 0)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", False)
    args.use_fine_model = getattr(args, "use_fine_model", False)
    sdf_nsvf_architecture(args)

@register_model_architecture("sdf_nerf", "sdf_nerf_coarse_64")
def sdf_nsvf3_architecture(args):
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:6:3")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, normal:0:3, pos:0:3, ray:4:3:b")
    sdf_nsvf2_architecture(args)


@register_model('sg_nerf')
class SGNeRFModel(NeRFModel):
    """ This is a simple re-implementation of the vanilla NeRF
    """
    ENCODER = 'infinite_volume_encoder'

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # vanilla nerf hits everything. so no need to fill_in
        S, V, P = sizes
        all_results['missed'] = all_results['missed'].view(S, V, P)
        all_results['colors'] = all_results['colors'].view(S, V, P, 3)
        all_results['depths'] = all_results['depths'].view(S, V, P)
        if 'z' in all_results:
            all_results['z'] = all_results['z'].view(S, V, P)
        if 'normal' in all_results:
            all_results['normal'] = all_results['normal'].view(S, V, P, 3)
        return all_results

@register_model_architecture("sg_nerf", "sg_nerf_base")
def sg_nerf_architecture(args):
    INF_FAR = 1e6
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10:4")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4:3:b")
    args.near = getattr(args, "near", 2)
    args.far = getattr(args, "far", INF_FAR)
    base_architecture(args)


@register_model_architecture("sg_nerf", "sg_nerf_new")
def sg_nerf2_architecture(args):
    args.nerf_style_mlp = getattr(args, "nerf_style_mlp", True)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 128)
    sg_nerf_architecture(args)


@register_model('joint_nerf')
class JointNeRFModel(NeRFModel):

    ENCODER = "joint_volume_encoder"
    # ENCODER = "joint_volume_encoder2"

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--skip-rendering', action='store_true')
        parser.add_argument('--black-background', action='store_true')

    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        # FIXME later
        if not getattr(self.args, "skip_rendering", False):
            samples, all_results = super().raymarching(ray_start, ray_dir, intersection_outputs, encoder_states, fine)
        else:
            samples, all_results = {}, {}
            
        all_results['voxel_depth'] = intersection_outputs['mesh_depth']
        all_results['voxel_masks'] = intersection_outputs['mesh_hits']
        all_results['voxel_edges'] = intersection_outputs['mesh_color']
        # all_results['voxel_edges'] = self.encoder.get_edge(ray_start, ray_dir, samples, encoder_states)
        
        if intersection_outputs.get('skel_depth', None) is not None:
            all_results['skel_depth'] = intersection_outputs['skel_depth']
            all_results['skel_masks'] = intersection_outputs['skel_hits']
            all_results['skel_edges'] = self.encoder.get_edge(ray_start, ray_dir, samples, encoder_states)
        return samples, all_results

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
         # we need fill_in for NSVF for background
        S, V, P = sizes
        fullsize = S * V * P
        BG_DEPTH = self.field.bg_color.depth
        device = ray_start.device
        if 'missed' in all_results:
            all_results['missed'] = fill_in((fullsize, ), hits, all_results['missed'], 1.0, device).view(S, V, P)
            all_results['depths'] = fill_in((fullsize, ), hits, all_results['depths'], 0.0, device).view(S, V, P)
            all_results['depths'] += all_results['missed'] * BG_DEPTH
            all_results['colors'] = fill_in((fullsize, 3), hits, all_results['colors'], 0.0, device).view(S, V, P, 3)
            bg_color = self.field.bg_color(all_results['colors'])
            if getattr(self.args, "black_background", False):
                bg_color = bg_color * 0.0
            all_results['colors'] += all_results['missed'].unsqueeze(-1) * bg_color.reshape(fullsize, 3).view(S, V, P, 3)
        else:
            all_results['colors'] = ray_start.new_ones(S, V, P, 3)

        if 'normal' in all_results:
            all_results['normal'] = fill_in((fullsize, 3), hits, all_results['normal'], 0.0, device).view(S, V, P, 3)
        if 'feat_n2' in all_results:
            all_results['feat_n2'] = fill_in((fullsize,), hits, all_results['feat_n2'], 0.0, device).view(S, V, P)
        if 'joint_colors' in all_results:
            all_results['joint_colors'] = fill_in((fullsize, 4), hits, all_results['joint_colors'], 0.0, device).view(S, V, P, 4)

        if 'voxel_masks' in all_results:
            all_results['voxel_depth'] = all_results['voxel_depth'].masked_fill_(~all_results['voxel_masks'], BG_DEPTH)
            all_results['voxel_edges'] = all_results['voxel_edges'].masked_fill_(~all_results['voxel_masks'][:, None], 
                0.0 if getattr(self.args, "black_background", False) else 1.0)
            all_results['voxel_depth'] = fill_in((fullsize, ), hits, all_results['voxel_depth'], BG_DEPTH, device).view(S, V, P)
            all_results['voxel_edges'] = fill_in((fullsize, 3), hits, all_results['voxel_edges'], 
                0.0 if getattr(self.args, "black_background", False) else 1.0, device).view(S, V, P, 3)

        if 'skel_masks' in all_results:
            all_results['skel_depth'] = all_results['skel_depth'].masked_fill_(~all_results['skel_masks'], BG_DEPTH)
            all_results['skel_edges'] = all_results['skel_edges'].masked_fill_(~all_results['skel_masks'][:, None], 1.0)
            all_results['skel_depth'] = fill_in((fullsize, ), hits, all_results['skel_depth'], BG_DEPTH, device).view(S, V, P)
            all_results['skel_edges'] = fill_in((fullsize, 3), hits, all_results['skel_edges'], 1.0, device).view(S, V, P, 3)

        return all_results

    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state
        images = super()._visualize(images, sample, output, state, **kwargs)
        if 'joint_colors' in output and output['joint_colors'] is not None:
            images['{}_jcolors/{}:HWC'.format(name, img_id)] = {
                'img': output['joint_colors'][shape, view].float(),
                'min_val': 0,
                'max_val': 1
            }
        if 'voxel_edges' in output and output['voxel_edges'] is not None:
            # voxel hitting visualization
            images['{}_voxel/{}:HWC'.format(name, img_id)] = {
                'img': output['voxel_edges'][shape, view].float(), 
                'min_val': 0, 
                'max_val': 1,
                # 'weight':
                #     compute_normal_map(
                #         sample['ray_start'][shape, view].float(),
                #         sample['ray_dir'][shape, view].float(),
                #         output['voxel_depth'][shape, view].float(),
                #         sample['extrinsics'][shape, view].float().inverse(),
                #         width, proj=True)
                }
        
        if 'skel_edges' in output and output['skel_edges'] is not None:
            # voxel hitting visualization
            images['{}_skel/{}:HWC'.format(name, img_id)] = {
                'img': output['skel_edges'][shape, view].float(), 
                'min_val': 0, 
                'max_val': 1,
                'weight':
                    compute_normal_map(
                        sample['ray_start'][shape, view].float(),
                        sample['ray_dir'][shape, view].float(),
                        output['skel_depth'][shape, view].float(),
                        sample['extrinsics'][shape, view].float().inverse(),
                        width, proj=True)
                }

        if hasattr(self.encoder, "texture_mask") and self.encoder.texture_mask is not None:
            images['{}_texmask/{}:HWC'.format(name, img_id)] = {
                'img': torch.sigmoid(self.encoder.texture_mask[..., None]),
                'min_val': 0,
                'max_val': 1
            }

        return images


@register_model_architecture("joint_nerf", "joint_nerf_coarse")
def joint_nerf_coarse_architecture(args):
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 80)
    args.joint_dropout = getattr(args, "joint_dropout", 0.0)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 0)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", False)
    args.use_fine_model = getattr(args, "use_fine_model", False)
    args.texture_layers = getattr(args, "texture_layers", 1)
    args.disable_guided_skinning = getattr(args, "disable_guided_skinning", False)
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:6:3")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, attnout:0:256, ray:4:3:b")
    nerf_nerf_architecture(args)

@register_model_architecture("joint_nerf", "joint_nerf_coarse_context")
def joint_nerf_coarse_context_architecture(args):
    args.joint_feature_dim = getattr(args, "joint_feature_dim", 256)
    args.trainable_object_embedding = getattr(args, "trainable_object_embedding", "(64,19500)")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, attnout:0:320, ray:4:3:b")
    joint_nerf_coarse_architecture(args)

@register_model_architecture("joint_nerf", "joint_nerf_coarse_experts")
def joint_nerf_coarse_context_architecture(args):
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:6:3")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, attnout:0:256, ray:4:3:b")
    joint_nerf_coarse_architecture(args)

@register_model_architecture("joint_nerf", "joint_nerf_baseline")
def joint_nerf_baseline_architecture(args):
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 80)
    args.joint_dropout = getattr(args, "joint_dropout", 0.0)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 0)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", False)
    args.use_fine_model = getattr(args, "use_fine_model", False)
    args.texture_layers = getattr(args, "texture_layers", 1)
    args.disable_guided_skinning = getattr(args, "disable_guided_skinning", True)
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:6:3")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, attnout:0:256, ray:4:3:b")
    args.disable_transformer = getattr(args, "disable_transformer", True)
    nerf_nerf_architecture(args)

@register_model_architecture("joint_nerf", "joint_nerf_direct")
def joint_nerf_direct_architecture(args):
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 80)
    args.joint_dropout = getattr(args, "joint_dropout", 0.0)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 0)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", False)
    args.use_fine_model = getattr(args, "use_fine_model", False)
    args.texture_layers = getattr(args, "texture_layers", 1)
    args.disable_projection = getattr(args, "disable_projection", True)
    args.disable_guided_skinning = getattr(args, "disable_guided_skinning", True)
    args.inputs_to_density = getattr(args, "inputs_to_density", "attnout:0:256")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4:3:b")
    nerf_nerf_architecture(args)

@register_model_architecture("joint_nerf", "joint_nerf")
def joint_nerf_architecture(args):
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 40)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 120)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", True)
    args.use_fine_model = getattr(args, "use_fine_model", True)
    args.texture_layers = getattr(args, "texture_layers", 1)
    args.disable_guided_skinning = getattr(args, "disable_guided_skinning", False)
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:6:3")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, attnout:0:256, ray:4:3:b")
    nerf_nerf_architecture(args)