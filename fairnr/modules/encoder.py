# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import open3d as o3d
import numpy as np
import math
import sys

import os
import math
import json
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from fairnr.data.data_utils import load_matrix
from fairnr.data import geometry
from fairnr.data import mesh as mesh_ops
from fairnr.data.geometry import (
    trilinear_interp, splitting_points, offset_points,
    get_edge, build_easy_octree, discretize_points,
    barycentric_interp
)
from fairnr.clib import (
    aabb_ray_intersect, triangle_ray_intersect, svo_ray_intersect,
    uniform_ray_sampling, inverse_cdf_sampling
)
from fairnr.modules.module_utils import (
    FCBlock, Linear, Embedding,
    InvertableMapping,
    NeRFPosEmbLinear
)
MAX_DEPTH = 10000.0
ENCODER_REGISTRY = {}

def register_encoder(name):
    def register_encoder_cls(cls):
        if name in ENCODER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        ENCODER_REGISTRY[name] = cls
        return cls
    return register_encoder_cls


def get_encoder(name):
    if name not in ENCODER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return ENCODER_REGISTRY[name]


@register_encoder('abstract_encoder')
class Encoder(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass


@register_encoder('volume_encoder')
class VolumeEncoder(Encoder):
    
    def __init__(self, args):
        super().__init__(args)

        self.context = None
        self.near = args.near
        self.far = args.far

    @staticmethod
    def add_args(parser):
        parser.add_argument('--near', type=float, help='near distance of the volume')
        parser.add_argument('--far',  type=float, help='far distance of the volume')

    def precompute(self, id=None, context=None, *args, **kwargs):
        self.context = context  # save context which maybe useful later
        return {}   # we do not use encoder for NeRF

    def ray_intersect(self, ray_start, ray_dir, encoder_states, near=None, far=None):
        assert getattr(self.args, "fixed_num_samples", None) is not None

        S, V, P, _ = ray_dir.size()
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        near = near if near is not None else self.args.near
        far = far if far is not None else self.args.far
        intersection_outputs = {
            "min_depth": ray_dir.new_ones(S, V * P, 1) * near,
            "max_depth": ray_dir.new_ones(S, V * P, 1) * far,
            "probs": ray_dir.new_ones(S, V * P, 1),
            "steps": ray_dir.new_ones(S, V * P) * self.args.fixed_num_samples,
            "intersected_voxel_idx": ray_dir.new_zeros(S, V * P, 1).int()}
        hits = ray_dir.new_ones(S, V * P).bool()
        return ray_start, ray_dir, intersection_outputs, hits

    def ray_sample(self, intersection_outputs):
        sampled_idx, sampled_depth, sampled_dists = inverse_cdf_sampling(
            intersection_outputs['intersected_voxel_idx'], 
            intersection_outputs['min_depth'], 
            intersection_outputs['max_depth'], 
            intersection_outputs['probs'],
            intersection_outputs['steps'], -1, (not self.training))
        return {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,  # dummy index (to match raymarcher)
        }

    def forward(self, samples, encoder_states):
        inputs = {
            'pos': samples['sampled_point_xyz'].requires_grad_(True),
            'ray': samples.get('sampled_point_ray_direction', None),
            'dists': samples.get('sampled_point_distance', None)
        }
        if self.context is not None:
            inputs.update({'context': self.context})
        return inputs

    def generate_random_samples(self, num_samples):
        # Sample points for the eikonal loss
        eik_bounding_box = (self.far - self.near) / 2.
        eikonal_points = torch.empty(num_samples, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
        eikonal_points = self.forward({'sampled_point_xyz': eikonal_points})
        return eikonal_points
        
    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        # Token from IDR code.
        from skimage import measure

        def get_grid_uniform(resolution):
            x = np.linspace(-2.0, 2.0, resolution)
            xx, yy, zz = np.meshgrid(x, x, x)
            grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

            return {"grid_points": grid_points.cuda(),
                    "shortest_axis_length": 2.0,
                    "xyz": [x, x, x],
                    "shortest_axis_index": 0}

        grid = get_grid_uniform(resolution=400)
        points = grid['grid_points']
        z = []
        for pnts in tqdm(torch.split(points, 100000, dim=0)):
            z.append(field_fn({'pos': pnts}, outputs=['sigma', 'sdf'])['sdf'].detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (not (np.min(z) > 0 or np.max(z) < 0)):
            z = z.astype(np.float32)
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=0,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
            verts = np.array([tuple(a) for a in verts.tolist()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            faces = np.array([(a, ) for a in faces.tolist()], dtype=[('vertex_indices', 'i4', (3,))])
            return PlyData([PlyElement.describe(verts, 'vertex'), PlyElement.describe(faces, 'face')])
        raise NotImplementedError


@register_encoder('infinite_volume_encoder')
class InfiniteVolumeEncoder(VolumeEncoder):

    def __init__(self, args):
        super().__init__(args)
        self.imap = InvertableMapping(style='simple')
        self.nofixdz = getattr(args, "no_fix_dz", False)
        self.sample_msi = getattr(args, "sample_msi", False)

    @staticmethod
    def add_args(parser):
        VolumeEncoder.add_args(parser)
        parser.add_argument('--no-fix-dz', action='store_true', help='do not fix dz.')
        parser.add_argument('--sample-msi', action='store_true')

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        S, V, P, _ = ray_dir.size()
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        
        # ray sphere (unit) intersection (assuming all camera is inside sphere):
        p_v = (ray_start * ray_dir).sum(-1)
        p_p = (ray_start * ray_start).sum(-1)
        d_u = -p_v + torch.sqrt(p_v ** 2 - p_p + 1)
        
        intersection_outputs = {
            "min_depth": torch.arange(-1, 1, 1, dtype=ray_dir.dtype, device=ray_dir.device)[None, None, :].expand(S, V * P, 2),
            "max_depth": torch.arange( 0, 2, 1, dtype=ray_dir.dtype, device=ray_dir.device)[None, None, :].expand(S, V * P, 2),
            "probs": ray_dir.new_ones(S, V * P, 2) * .5,
            "steps": ray_dir.new_ones(S, V * P, 1) * self.args.fixed_num_samples,
            "intersected_voxel_idx": torch.arange( 0, 2, 1, device=ray_dir.device)[None, None, :].expand(S, V * P, 2).int(),
            "unit_sphere_depth": d_u,
            "p_v": p_v, "p_p": p_p}
        hits = ray_dir.new_ones(S, V * P).bool()
        return ray_start, ray_dir, intersection_outputs, hits
        
    def ray_sample(self, intersection_outputs):
        samples = super().ray_sample(intersection_outputs)   # HACK: < 1, unit sphere;  > 1, outside the sphere
        
        # map from (0, 1) to (0, +inf) with invertable mapping
        samples['original_point_distance'] = samples['sampled_point_distance'].clone()
        samples['original_point_depth'] = samples['sampled_point_depth'].clone()
        
        # assign correct depth
        in_depth = intersection_outputs['unit_sphere_depth'][:, None] * (
            samples['original_point_depth'].clamp(max=0.0) + 1.0).masked_fill(samples['sampled_point_voxel_idx'].ne(0), 0)
        if not self.sample_msi:
            out_depth = (intersection_outputs['unit_sphere_depth'][:, None] + 1 / (1 - samples['original_point_depth'].clamp(min=0.0) + 1e-7) - 1
                ).masked_fill(samples['sampled_point_voxel_idx'].ne(1), 0)
        else:
            p_v, p_p = intersection_outputs['p_v'][:, None], intersection_outputs['p_p'][:, None]
            out_depth = (-p_v + torch.sqrt(p_v ** 2 - p_p + 1. / (1. - samples['original_point_depth'].clamp(min=0.0) + 1e-7) ** 2)
                ).masked_fill(samples['sampled_point_voxel_idx'].ne(1), 0)
        samples['sampled_point_depth'] = in_depth + out_depth

        if not self.nofixdz:
            # raise NotImplementedError("need to re-compute later")
            in_dists = 1 / intersection_outputs['unit_sphere_depth'][:, None] * (samples['original_point_distance']).masked_fill(
                samples['sampled_point_voxel_idx'].ne(0), 0)
            alpha = 1. if not self.sample_msi else 1. / torch.sqrt(1. + (p_v ** 2 - p_p) * (1. - samples['original_point_depth'].clamp(min=0.0) + 1e-7) ** 2)
            out_dists = alpha / ((1 - samples['original_point_depth'].clamp(min=0.0)) ** 2 + 1e-7) * (samples['original_point_distance']).masked_fill(
                samples['sampled_point_voxel_idx'].ne(1), 0)
            samples['sampled_point_distance'] = in_dists + out_dists
        else:
            samples['sampled_point_distance'] = samples['sampled_point_distance'].scatter(1, 
                samples['sampled_point_voxel_idx'].ne(-1).sum(-1, keepdim=True) - 1, 1e8)
        
        return samples

    def forward(self, samples, encoder_states):
        field_inputs = super().forward(samples, encoder_states)

        r = field_inputs['pos'].norm(p=2, dim=-1, keepdim=True) # .clamp(min=1.0)
        field_inputs['pos'] = torch.cat([field_inputs['pos'] / (r + 1e-8), r / (1.0 + r)], dim=-1)
        return field_inputs


@register_encoder('sparsevoxel_encoder')
class SparseVoxelEncoder(Encoder):

    def __init__(self, args, voxel_path=None, bbox_path=None, shared_values=None):
        super().__init__(args)
        # read initial voxels or learned sparse voxels
        self.voxel_path = voxel_path if voxel_path is not None else args.voxel_path
        self.bbox_path = bbox_path if bbox_path is not None else getattr(args, "initial_boundingbox", None)
        assert (self.bbox_path is not None) or (self.voxel_path is not None), \
            "at least initial bounding box or pretrained voxel files are required."
        self.voxel_index = None
        self.scene_scale = getattr(args, "scene_scale", 1.0)

        if self.voxel_path is not None:
            # read voxel file
            assert os.path.exists(self.voxel_path), "voxel file must exist"
            
            if Path(self.voxel_path).suffix == '.ply':
                from plyfile import PlyData, PlyElement
                plyvoxel = PlyData.read(self.voxel_path)
                elements = [x.name for x in plyvoxel.elements]
                
                assert 'vertex' in elements
                plydata = plyvoxel['vertex']
                fine_points = torch.from_numpy(
                    np.stack([plydata['x'], plydata['y'], plydata['z']]).astype('float32').T)

                if 'face' in elements:
                    # read voxel meshes... automatically detect voxel size
                    faces = plyvoxel['face']['vertex_indices']
                    t = fine_points[faces[0].astype('int64')]
                    voxel_size = torch.abs(t[0] - t[1]).max()

                    # indexing voxel vertices
                    fine_points = torch.unique(fine_points, dim=0)

                else:
                    # voxel size must be provided
                    assert getattr(args, "voxel_size", None) is not None, "final voxel size is essential."
                    voxel_size = args.voxel_size

                if 'quality' in elements:
                    self.voxel_index = torch.from_numpy(plydata['quality']).long()
               
            else:
                # supporting the old style .txt voxel points
                fine_points = torch.from_numpy(np.loadtxt(self.voxel_path)[:, 3:].astype('float32'))
        else:
            # read bounding-box file
            bbox = np.loadtxt(self.bbox_path)
            voxel_size = bbox[-1] if getattr(args, "voxel_size", None) is None else args.voxel_size
            fine_points = torch.from_numpy(bbox2voxels(bbox[:6], voxel_size))
        
        half_voxel = voxel_size * .5
        
        # transform from voxel centers to voxel corners (key/values)
        fine_coords, _ = discretize_points(fine_points, half_voxel)
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)
        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()
        
        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        # register parameters (will be saved to checkpoints)
        self.register_buffer("points", fine_points)          # voxel centers
        self.register_buffer("keys", fine_keys.long())       # id used to find voxel corners/embeddings
        self.register_buffer("feats", fine_feats.long())     # for each voxel, 8 voxel corner ids
        self.register_buffer("num_keys", num_keys)
        self.register_buffer("keep", fine_feats.new_ones(fine_feats.size(0)).long())  # whether the voxel will be pruned

        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        logger.info("loaded {} voxel centers, {} voxel corners".format(fine_points.size(0), num_keys))

        # set-up other hyperparameters and initialize running time caches
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)    
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        # sparse voxel embeddings     
        if shared_values is None and self.embed_dim > 0:
            self.values = Embedding(num_keys, self.embed_dim, None)
        else:
            self.values = shared_values

    def upgrade_state_dict_named(self, state_dict, name):
        # update the voxel embedding shapes
        if self.values is not None:
            loaded_values = state_dict[name + '.values.weight']
            self.values.weight = nn.Parameter(self.values.weight.new_zeros(*loaded_values.size()))
            self.values.num_embeddings = self.values.weight.size(0)
            self.total_size = self.values.weight.size(0)
            self.num_keys = self.num_keys * 0 + self.total_size
        
        if self.voxel_index is not None:
            state_dict[name + '.points'] = state_dict[name + '.points'][self.voxel_index]
            state_dict[name + '.feats'] = state_dict[name + '.feats'][self.voxel_index]
            state_dict[name + '.keep'] = state_dict[name + '.keep'][self.voxel_index]
        
        # update the buffers shapes
        if name + '.points' in state_dict:
            self.points = self.points.new_zeros(*state_dict[name + '.points'].size())
            self.feats  = self.feats.new_zeros(*state_dict[name + '.feats'].size())
            self.keys   = self.keys.new_zeros(*state_dict[name + '.keys'].size())
            self.keep   = self.keep.new_zeros(*state_dict[name + '.keep'].size())
        
        else:
            # this usually happens when loading a NeRF checkpoint to NSVF
            # use initialized values
            state_dict[name + '.points'] = self.points
            state_dict[name + '.feats'] = self.feats
            state_dict[name + '.keys'] = self.keys
            state_dict[name + '.keep'] = self.keep
    
            state_dict[name + '.voxel_size'] = self.voxel_size
            state_dict[name + '.step_size'] = self.step_size
            state_dict[name + '.max_hits'] = self.max_hits
            state_dict[name + '.num_keys'] = self.num_keys

    @staticmethod
    def add_args(parser):
        parser.add_argument('--initial-boundingbox', type=str, help='the initial bounding box to initialize the model')
        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial')
        parser.add_argument('--voxel-path', type=str, help='path for pretrained voxel file. if provided no update')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-stepsize-ratio', type=float, metavar='D',
                            help='if the concrete step size is not given (=0), we use the ratio to the voxel size as step size.')
        parser.add_argument('--use-octree', action='store_true', help='if set, instead of looping over the voxels, we build an octree.')
        parser.add_argument('--track-max-probs', action='store_true', help='if set, tracking the maximum probability in ray-marching.')
        parser.add_argument('--scene-scale', type=float, default=1.0)

    def reset_runtime_caches(self):
        logger.info("reset chache")
        if self.use_octree:
            points = self.points[self.keep.bool()]
            centers, children = build_easy_octree(points, self.voxel_size / 2.0)
            self._runtime_caches['flatten_centers'] = centers
            self._runtime_caches['flatten_children'] = children
        if self.track_max_probs:
            self._runtime_caches['max_voxel_probs'] = self.points.new_zeros(self.points.size(0))

    def clean_runtime_caches(self):
        logger.info("clean chache")
        for name in self._runtime_caches:
            self._runtime_caches[name] = None

    def precompute(self, id=None, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        points[:, 0] += (self.voxel_size / 10)
        values = self.values.weight[: self.num_keys] if self.values is not None else None
        
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if self.use_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            if id is not None:
                flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
                flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children
        return encoder_states

    @torch.no_grad()
    def export_voxels(self, return_mesh=False):
        logger.info("exporting learned sparse voxels...")
        voxel_idx = torch.arange(self.keep.size(0), device=self.keep.device)
        voxel_idx = voxel_idx[self.keep.bool()]
        voxel_pts = self.points[self.keep.bool()]
        if not return_mesh:
            # HACK: we export the original voxel indices as "quality" in case for editing
            points = [
                (voxel_pts[k, 0], voxel_pts[k, 1], voxel_pts[k, 2], voxel_idx[k])
                for k in range(voxel_idx.size(0))
            ]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])
            return PlyData([PlyElement.describe(vertex, 'vertex')])
        
        else:
            # generate polygon for voxels
            center_coords, residual = discretize_points(voxel_pts, self.voxel_size / 2)
            offsets = torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1]], device=center_coords.device)
            vertex_coords = center_coords[:, None, :] + offsets[None, :, :]
            vertex_points = vertex_coords.type_as(residual) * self.voxel_size / 2 + residual
            
            faceidxs = [[1,6,7,5],[7,6,2,4],[5,7,4,3],[1,0,2,6],[1,5,3,0],[0,3,4,2]]
            all_vertex_keys, all_vertex_idxs  = {}, []
            for i in range(vertex_coords.shape[0]):
                for j in range(8):
                    key = " ".join(["{}".format(int(p)) for p in vertex_coords[i,j]])
                    if key not in all_vertex_keys:
                        all_vertex_keys[key] = vertex_points[i,j]
                        all_vertex_idxs += [key]
            all_vertex_dicts = {key: u for u, key in enumerate(all_vertex_idxs)}
            all_faces = torch.stack([torch.stack([vertex_coords[:, k] for k in f]) for f in faceidxs]).permute(2,0,1,3).reshape(-1,4,3)
    
            all_faces_keys = {}
            for l in range(all_faces.size(0)):
                key = " ".join(["{}".format(int(p)) for p in all_faces[l].sum(0) // 4])
                if key not in all_faces_keys:
                    all_faces_keys[key] = all_faces[l]

            vertex = np.array([tuple(all_vertex_keys[key].cpu().tolist()) for key in all_vertex_idxs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            face = np.array([([all_vertex_dicts["{} {} {}".format(*b)] for b in a.cpu().tolist()],) for a in all_faces_keys.values()],
                dtype=[('vertex_indices', 'i4', (4,))])
            return PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])

    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        """
        extract triangle-meshes from the implicit field using marching cube algorithm
            Lewiner, Thomas, et al. "Efficient implementation of marching cubes' cases with topological guarantees." 
            Journal of graphics tools 8.2 (2003): 1-15.
        """
        logger.info("marching cube...")
        encoder_states = self.precompute(id=None)
        points = encoder_states['voxel_center_xyz']

        scores = self.get_scores(field_fn, th=th, bits=bits, encoder_states=encoder_states)
        coords, residual = discretize_points(points, self.voxel_size)
        A, B, C = [s + 1 for s in coords.max(0).values.cpu().tolist()]
    
        # prepare grids
        full_grids = points.new_ones(A * B * C, bits ** 3)
        full_grids[coords[:, 0] * B * C + coords[:, 1] * C + coords[:, 2]] = scores
        full_grids = full_grids.reshape(A, B, C, bits, bits, bits)
        full_grids = full_grids.permute(0, 3, 1, 4, 2, 5).reshape(A * bits, B * bits, C * bits)
        full_grids = 1 - full_grids

        # marching cube
        from skimage import measure
        space_step = self.voxel_size.item() / bits
        verts, faces, normals, _ = measure.marching_cubes_lewiner(
            volume=full_grids.cpu().numpy(), level=0.0, # 0.5
            spacing=(space_step, space_step, space_step)
        )
        verts += (residual - (self.voxel_size / 2)).cpu().numpy()
        verts = np.array([tuple(a) for a in verts.tolist()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        faces = np.array([(a, ) for a in faces.tolist()], dtype=[('vertex_indices', 'i4', (3,))])
        return PlyData([PlyElement.describe(verts, 'vertex'), PlyElement.describe(faces, 'face')])

    def get_edge(self, ray_start, ray_dir, samples, encoder_states):
        outs = get_edge(
            ray_start + ray_dir * samples['sampled_point_depth'][:, :1], 
            encoder_states['voxel_center_xyz'].reshape(-1, 3)[samples['sampled_point_voxel_idx'][:, 0].long()], 
            self.voxel_size).type_as(ray_dir)   # get voxel edges/depth (for visualization)
        outs = (1 - outs[:, None].expand(outs.size(0), 3)) * 0.7
        return outs

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()

        if self.use_octree:  # ray-voxel intersection with SVO
            flatten_centers = encoder_states['voxel_octree_center_xyz']
            flatten_children = encoder_states['voxel_octree_children_idx']
            pts_idx, min_depth, max_depth = svo_ray_intersect(
                self.voxel_size, self.max_hits, flatten_centers, flatten_children,
                ray_start, ray_dir)
        else:   # ray-voxel intersection with all voxels
            pts_idx, min_depth, max_depth = aabb_ray_intersect(
                self.voxel_size, self.max_hits, point_xyz, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        
        if S > 1:  # extend the point-index to multiple shapes (just in case)
            pts_idx = (pts_idx + H * torch.arange(S, 
                device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                ).masked_fill_(pts_idx.eq(-1), -1)

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits

    def ray_sample(self, intersection_outputs):
        # sample points and use middle point approximation
        sampled_idx, sampled_depth, sampled_dists = inverse_cdf_sampling(
            intersection_outputs['intersected_voxel_idx'],
            intersection_outputs['min_depth'], 
            intersection_outputs['max_depth'], 
            intersection_outputs['probs'],
            intersection_outputs['steps'], 
            -1, self.deterministic_step or (not self.training))
        sampled_dists = sampled_dists.clamp(min=0.0)
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)
        
        samples = {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,
        }
        return samples

    @torch.enable_grad()
    def forward(self, samples, encoder_states=None):
        # ray point samples
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        sampled_xyz = samples['sampled_point_xyz'].requires_grad_(True)
        sampled_dir = samples.get('sampled_point_ray_direction', None)
        sampled_dis = samples.get('sampled_point_distance', None)

        # prepare inputs for implicit field
        inputs = {
            'pos': sampled_xyz, 
            'ray': sampled_dir, 
            'dists': sampled_dis}

        if self.values is not None:
            # encoder states
            point_feats = encoder_states['voxel_vertex_idx'] 
            point_xyz = encoder_states['voxel_center_xyz']
            values = encoder_states['voxel_vertex_emb']

            # resample point features
            point_xyz = F.embedding(sampled_idx, point_xyz)
            point_feats = F.embedding(F.embedding(sampled_idx, point_feats), values).view(point_xyz.size(0), -1)

            # tri-linear interpolation
            p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
            q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5  # BUG (FIX) 
            inputs.update({'emb': trilinear_interp(p, q, point_feats)})

        return inputs

    @torch.no_grad()
    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        voxel_idxs = voxel_idxs.masked_fill(voxel_idxs.eq(-1), self.max_voxel_probs.size(0))
        chunk_size = 4096
        for start in range(0, voxel_idxs.size(0), chunk_size):
            end = start + chunk_size
            end = end if end < voxel_idxs.size(0) else voxel_idxs.size(0)
            max_voxel_probs = self.max_voxel_probs.new_zeros(end-start, self.max_voxel_probs.size(0) + 1).scatter_add_(
                dim=-1, index=voxel_idxs[start:end], src=voxel_probs[start:end]).max(0)[0][:-1].data        
            self.max_voxel_probs = torch.max(self.max_voxel_probs, max_voxel_probs)
    
    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, encoder_states=None, train_stats=False):
        if not train_stats:
            logger.info("pruning...")
            scores = self.get_scores(field_fn, th=th, bits=16, encoder_states=encoder_states)
            keep = (1 - scores.min(-1)[0]) > th
        else:
            logger.info("pruning based on training set statics (e.g. probs)...")
            if dist.is_initialized() and dist.get_world_size() > 1:  # sync on multi-gpus
                dist.all_reduce(self.max_voxel_probs, op=dist.ReduceOp.MAX)
            keep = self.max_voxel_probs > th
            
        self.keep.masked_scatter_(self.keep.bool(), keep.long())
        logger.info("pruning done. # of voxels before: {}, after: {} voxels".format(keep.size(0), keep.sum()))
    
    def get_scores(self, field_fn, th=0.5, bits=16, encoder_states=None):
        if encoder_states is None:
            encoder_states = self.precompute(id=None)
        
        feats = encoder_states['voxel_vertex_idx'] 
        points = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']
        chunk_size = 64

        def get_scores_once(feats, points, values):
            # sample points inside voxels
            sampled_xyz = offset_points(points, self.voxel_size / 2.0, bits=bits)
            sampled_idx = torch.arange(points.size(0), device=points.device)[:, None].expand(*sampled_xyz.size()[:2])
            sampled_xyz, sampled_idx = sampled_xyz.reshape(-1, 3), sampled_idx.reshape(-1)
            
            field_inputs = self.forward(
                {'sampled_point_xyz': sampled_xyz, 
                 'sampled_point_voxel_idx': sampled_idx,
                 'sampled_point_ray_direction': None,
                 'sampled_point_distance': None}, 
                {'voxel_vertex_idx': feats,
                 'voxel_center_xyz': points,
                 'voxel_vertex_emb': values})  # get field inputs
            if encoder_states.get('context', None) is not None:
                field_inputs['context'] = encoder_states['context']
            
            # evaluation with density
            field_outputs = field_fn(field_inputs, outputs=['sigma', 'sdf'])
            free_energy = -torch.relu(field_outputs['sigma']).reshape(-1, bits ** 3)
            return torch.exp(free_energy)

        return torch.cat([get_scores_once(feats[i: i + chunk_size], points[i: i + chunk_size], values) 
            for i in range(0, points.size(0), chunk_size)], 0)

    def generate_random_samples(self, num_samples):
        # Sample points inside voxels
        centers = self.points[self.keep.bool()]
        feats  = self.feats[self.keep.bool()]
        halfsize = self.voxel_size

        sampled_indices = torch.empty(num_samples).uniform_(0., float(centers.size(0))).type_as(centers).long()
        sampled_centers = F.embedding(sampled_indices, centers)
        eikonal_points = torch.empty(num_samples, 3).uniform_(-halfsize, halfsize).type_as(centers)
        eikonal_points = eikonal_points + sampled_centers
        eikonal_feats = F.embedding(sampled_indices, feats)
        eikonal_points = self.forward({
            'sampled_point_xyz': eikonal_points,
            'sampled_point_voxel_idx': eikonal_feats})
        return eikonal_points

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        encoder_states = self.precompute(id=None)
        feats, points, values = encoder_states['voxel_vertex_idx'], encoder_states['voxel_center_xyz'], encoder_states['voxel_vertex_emb']
        new_points, new_feats, new_values, new_keys = splitting_points(points, feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)
        
        # set new voxel embeddings
        if new_values is not None:
            self.values.weight = nn.Parameter(new_values)
            self.values.num_embeddings = self.values.weight.size(0)
        
        self.total_size = new_num_keys
        self.num_keys = self.num_keys * 0 + self.total_size

        self.points = new_points
        self.feats = new_feats
        self.keep = self.keep.new_ones(new_point_length)
        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(points.size(0), self.keep.sum()))
        
    @property
    def flatten_centers(self):
        if self._runtime_caches['flatten_centers'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_centers']
    
    @property
    def flatten_children(self):
        if self._runtime_caches['flatten_children'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_children']

    @property
    def max_voxel_probs(self):
        if self._runtime_caches['max_voxel_probs'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['max_voxel_probs']

    @max_voxel_probs.setter
    def max_voxel_probs(self, x):
        self._runtime_caches['max_voxel_probs'] = x

    @property
    def feature_dim(self):
        return self.embed_dim

    @property
    def dummy_loss(self):
        if self.values is not None:
            return self.values.weight[0,0] * 0.0
        return 0.0
    
    @property
    def num_voxels(self):
        return self.keep.long().sum()


@register_encoder('multi_sparsevoxel_encoder')
class MultiSparseVoxelEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        try:
            self.all_voxels = nn.ModuleList(
                [SparseVoxelEncoder(args, vox.strip()) for vox in open(args.voxel_path).readlines()])

        except TypeError:
            bbox_path = getattr(args, "bbox_path", "/private/home/jgu/data/shapenet/disco_dataset/bunny_point.txt")
            self.all_voxels = nn.ModuleList(
                [SparseVoxelEncoder(args, None, g.strip() + '/bbox.txt') for g in open(bbox_path).readlines()])
        
        # properties
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False) 

        self.cid = None
        if getattr(self.args, "global_embeddings", None) is not None:
            self.global_embed = torch.zeros(*eval(self.args.global_embeddings)).normal_(mean=0, std=0.01)
            self.global_embed = nn.Parameter(self.global_embed, requires_grad=True)
        else:
            self.global_embed = None

    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)
        parser.add_argument('--bbox-path', type=str, default=None)
        parser.add_argument('--global-embeddings', type=str, default=None,
            help="""set global embeddings if provided in global.txt. We follow this format:
                (N, D) or (K, N, D) if we have multi-dimensional global features. 
                D is the global feature dimentions. 
                N is the number of indices of this feature, 
                and K is the number of features if provided.""")

    def reset_runtime_caches(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].reset_runtime_caches()
    
    def clean_runtime_caches(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].clean_runtime_caches()

    def precompute(self, id, global_index=None, *args, **kwargs):
        # TODO: this is a HACK for simplicity
        assert id.size(0) == 1, "for now, only works for one object"
        
        # id = id * 0 + 2
        self.cid = id[0]
        encoder_states = self.all_voxels[id[0]].precompute(id, *args, **kwargs)
        if (global_index is not None) and (self.global_embed is not None):
            encoder_states['context'] = torch.stack([
                F.embedding(global_index[:, i], self.global_embed[i])
                for i in range(self.global_embed.size(0))], 1)
        return encoder_states

    def export_surfaces(self, field_fn, th, bits):
        raise NotImplementedError("does not support for now.")

    def export_voxels(self, return_mesh=False):
        raise NotImplementedError("does not support for now.")
    
    def get_edge(self, *args, **kwargs):
        return self.all_voxels[self.cid].get_edge(*args, **kwargs)

    def ray_intersect(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_intersect(*args, **kwargs)

    def ray_sample(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_sample(*args, **kwargs)

    def forward(self, samples, encoder_states):
        inputs = self.all_voxels[self.cid].forward(samples, encoder_states)
        if encoder_states.get('context', None) is not None:
            inputs['context'] = encoder_states['context']
        return inputs

    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        return self.all_voxels[self.cid].track_voxel_probs(voxel_idxs, voxel_probs)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, train_stats=False):
        for id in range(len(self.all_voxels)):
           self.all_voxels[id].pruning(field_fn, th, train_stats=train_stats)
    
    @torch.no_grad()
    def splitting(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].splitting()

    @property
    def feature_dim(self):
        return self.all_voxels[0].embed_dim

    @property
    def dummy_loss(self):
        return sum([d.dummy_loss for d in self.all_voxels])

    @property
    def voxel_size(self):
        return self.all_voxels[0].voxel_size

    @voxel_size.setter
    def voxel_size(self, x):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].voxel_size = x

    @property
    def step_size(self):
        return self.all_voxels[0].step_size

    @step_size.setter
    def step_size(self, x):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].step_size = x

    @property
    def max_hits(self):
        return self.all_voxels[0].max_hits

    @max_hits.setter
    def max_hits(self, x):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].max_hits = x

    @property
    def num_voxels(self):
        return self.all_voxels[self.cid].num_voxels


@register_encoder('shared_sparsevoxel_encoder')
class SharedSparseVoxelEncoder(MultiSparseVoxelEncoder):
    """
    Different from MultiSparseVoxelEncoder, we assume a shared list 
    of voxels across all models. Usually useful to learn a video sequence.  
    """
    def __init__(self, args):
        super(MultiSparseVoxelEncoder, self).__init__(args)

        # using a shared voxel
        self.voxel_path = args.voxel_path
        self.num_frames = args.num_frames
        self.all_voxels = [SparseVoxelEncoder(args, self.voxel_path)]
        self.all_voxels =  nn.ModuleList(self.all_voxels + [
            SparseVoxelEncoder(args, self.voxel_path, shared_values=self.all_voxels[0].values)
            for i in range(self.num_frames - 1)])
        self.context_embed_dim = args.context_embed_dim
        self.contexts = nn.Embedding(self.num_frames, self.context_embed_dim, None)
        self.cid = None

    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)
        parser.add_argument('--num-frames', type=int, help='the total number of frames')
        parser.add_argument('--context-embed-dim', type=int, help='context embedding for each view')

    def forward(self, samples, encoder_states):
        inputs = self.all_voxels[self.cid].forward(samples, encoder_states)
        inputs.update({'context': self.contexts(self.cid).unsqueeze(0)})
        return inputs

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, train_stats=False):
        for cid in range(len(self.all_voxels)):
           id = torch.tensor([cid], device=self.contexts.weight.device)
           encoder_states = {name: v[0] if v is not None else v 
                    for name, v in self.precompute(id).items()}
           encoder_states['context'] = self.contexts(id)
           self.all_voxels[cid].pruning(field_fn, th, 
                encoder_states=encoder_states,
                train_stats=train_stats)

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        all_feats, all_points = [], []
        for id in range(len(self.all_voxels)):
            encoder_states = self.all_voxels[id].precompute(id=None)
            feats = encoder_states['voxel_vertex_idx']
            points = encoder_states['voxel_center_xyz']
            values = encoder_states['voxel_vertex_emb']

            all_feats.append(feats)
            all_points.append(points)
        
        feats, points = torch.cat(all_feats, 0), torch.cat(all_points, 0)
        unique_feats, unique_idx = torch.unique(feats, dim=0, return_inverse=True)
        unique_points = points[
            unique_feats.new_zeros(unique_feats.size(0)).scatter_(
                0, unique_idx, torch.arange(unique_idx.size(0), device=unique_feats.device)
        )]
        new_points, new_feats, new_values, new_keys = splitting_points(unique_points, unique_feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)

        # set new voxel embeddings (shared voxels)
        if values is not None:
            self.all_voxels[0].values.weight = nn.Parameter(new_values)
            self.all_voxels[0].values.num_embeddings = new_num_keys

        for id in range(len(self.all_voxels)):
            self.all_voxels[id].total_size = new_num_keys
            self.all_voxels[id].num_keys = self.all_voxels[id].num_keys * 0 + self.all_voxels[id].total_size

            self.all_voxels[id].points = new_points
            self.all_voxels[id].feats = new_feats
            self.all_voxels[id].keep = self.all_voxels[id].keep.new_ones(new_point_length)

        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(
            unique_points.size(0), new_point_length))

    @property
    def feature_dim(self):
        return self.all_voxels[0].embed_dim + self.context_embed_dim


@register_encoder('triangle_mesh_encoder')
class TriangleMeshEncoder(SparseVoxelEncoder):
    """
    Training on fixed mesh model. Cannot pruning..
    """
    def __init__(self, args, mesh_path=None, shared_values=None):
        super(SparseVoxelEncoder, self).__init__(args)
        self.mesh_path = mesh_path if mesh_path is not None else args.mesh_path
        assert (self.mesh_path is not None) and os.path.exists(self.mesh_path)
        self.scale = getattr(args, "mesh_scale", 1.0)
        self.sample_center = getattr(args, "mesh_sample_triangle_center", False)
        self.gamma = getattr(args, "mesh_field_range", 0.24)
        self.sigma = getattr(args, "mesh_field_constant_density", 5.)
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        vertices, faces, sampled_triangles, sampled_points, sampled_normals, center, radius = \
            self.precompute_mesh(self.mesh)

        self.vertices = nn.Parameter(vertices, requires_grad=getattr(args, "trainable_vertices", False))
        self.faces = nn.Parameter(faces, requires_grad=False)
        
        # bounding sphere (R * 1.1 to make sure has some space)
        self.register_buffer("center", center)
        self.register_buffer("radius", radius)
        self.register_buffer("sampled_triangles", sampled_triangles)
        self.register_buffer("sampled_points", sampled_points)
        self.register_buffer("sampled_normals", sampled_normals)
        self.register_buffer("sampled_points_l2", (self.sampled_points ** 2).sum(-1, keepdim=True))
        
        self.ray_triangle_intersect = getattr(args, "ray_triangle_intersect", False)
        if self.ray_triangle_intersect:
            step_size = args.raymarching_stepsize
            if getattr(args, "raymarching_margin", None) is None:
                margin = step_size * 10  # truncated space around the triangle surfaces
            else:
                margin = args.raymarching_margin
            self.register_buffer("margin", torch.scalar_tensor(margin))
            self.register_buffer("step_size", torch.scalar_tensor(step_size))
            self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        # set-up other hyperparameters
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        if (self.embed_dim is not None) and (self.embed_dim > 0):
            # self.values = nn.Embedding(vertices.size(0), self.embed_dim)
            self.values = nn.Parameter(self.vertices.data, requires_grad=False)
            # self.values = nn.Parameter(torch.zeros(vertices.size(0), self.embed_dim).normal_(0.0, 0.02))
            # self.values.weight.data.copy_(self.vertices.data)
            # self.values.weight.requires_grad = False        
        else:
            self.values = None
        
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.blur_ratio = getattr(args, "blur_ratio", 0.0)

    def precompute_mesh(self, mesh, device='cpu', scale=None):
        scale = self.scale if scale is None else scale

        mesh.compute_triangle_normals()
        vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device) / scale
        faces = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.long)).to(device)
        sampled_triangles = F.embedding(faces, vertices)
        if not self.sample_center:
            samples = mesh.sample_points_uniformly(
                number_of_points=getattr(self.args, "mesh_sample_points", 5000),
                use_triangle_normal=True)
            sampled_points = torch.from_numpy(np.array(samples.points)).float().to(device) / scale
            sampled_normals = torch.from_numpy(np.array(samples.normals)).float().to(device)
        else:
            sampled_points = sampled_triangles.mean(1)
            sampled_normals = torch.from_numpy(np.array(mesh.triangle_normals)).float().to(device)
        center = vertices.mean(0)
        radius = (vertices - vertices.mean(0, keepdim=True)).norm(dim=1).max() * 1.1
        return vertices, faces, sampled_triangles, sampled_points, sampled_normals, center, radius

    def upgrade_state_dict_named(self, state_dict, name):
        pass
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--mesh-path', type=str, help='path for initial mesh file')
        parser.add_argument('--mesh-scale', type=float, default=1.)
        parser.add_argument('--mesh-sample-points', type=int, default=5000)
        parser.add_argument('--mesh-field-constant-density', type=float, default=5.)
        parser.add_argument('--mesh-field-range', type=float, default=0.24)
        parser.add_argument('--mesh-sample-triangle-center', action='store_true', help="use triangle centers as sampled points")

        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')

        parser.add_argument('--ray-triangle-intersect', action='store_true')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-margin', type=float, default=None,
                            help='margin around the surface.')
        parser.add_argument('--blur-ratio', type=float, default=0,
                            help="it is possible to shoot outside the triangle. default=0")
        
        parser.add_argument('--trainable-vertices', action='store_true',
                            help='if set, making the triangle trainable. experimental code. not ideal.')

    def precompute(self, id=None, vertex=None, *args, **kwargs):
        feats, points = self.faces, self.vertices
        if vertex is not None:
            self.mesh.vertices = o3d.utility.Vector3dVector(vertex[0].cpu().numpy())
            vertices, faces, sampled_triangles, sampled_points, sampled_normals, center, radius = \
                self.precompute_mesh(self.mesh, device=vertex.device, scale=1.0)
            
            # reset mesh propertices
            self.sampled_triangles = sampled_triangles
            self.sampled_points = sampled_points
            self.sampled_normals = sampled_normals
            self.sampled_points_l2 = (self.sampled_points ** 2).sum(-1, keepdim=True)
            self.center = center
            self.radius = radius
            
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            # values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None
            
            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + points.size(1) * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
    
        encoder_states = {
            'mesh_face_vertex_idx': feats,
            'mesh_vertex_xyz': points,
        }
        if 'all' in kwargs:
            encoder_states['all_images'] = kwargs['all'][0]
        return encoder_states

    def get_edge(self, ray_start, ray_dir, *args, **kwargs):
        return torch.ones_like(ray_dir) * 0.7

    @property
    def voxel_size(self):
        return self.margin

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        S, V, P, _ = ray_dir.size()
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()

        if not self.ray_triangle_intersect:
            assert getattr(self.args, "fixed_num_samples", None) is not None

            # we intersect the bounding sphere
            d_cet = self.center[None, None] - ray_start
            d_mid = (d_cet * ray_dir).sum(-1)
            d_dis = (d_cet ** 2).sum(-1) - d_mid ** 2

            hits = d_dis < self.radius ** 2
            d_dis = torch.sqrt((self.radius ** 2 - d_dis).clamp(min=0))
            min_depth = d_mid - d_dis
            max_depth = d_mid + d_dis

            intersection_outputs = {
                "min_depth": min_depth.unsqueeze(-1),
                "max_depth": max_depth.unsqueeze(-1),
                "probs": ray_dir.new_ones(S, V * P, 1),
                "intersected_voxel_idx": ray_dir.new_zeros(S, V * P, 1).int()}
        else:
            point_xyz = encoder_states['mesh_vertex_xyz']
            point_feats = encoder_states['mesh_face_vertex_idx']
            F, G = point_feats.size(1), point_xyz.size(1)
    
            # ray-voxel intersection
            pts_idx, depth, uv = triangle_ray_intersect(
                self.margin, self.blur_ratio, self.max_hits, point_xyz, point_feats, ray_start, ray_dir)
            min_depth = (depth[:,:,:,0] + depth[:,:,:,1]).masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
            max_depth = (depth[:,:,:,0] + depth[:,:,:,2]).masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
            hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object

            if S > 1:  # extend the point-index to multiple shapes (just in case)
                pts_idx = (pts_idx + G * torch.arange(S, 
                    device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                    ).masked_fill_(pts_idx.eq(-1), -1)

            intersection_outputs = {
                "min_depth": min_depth,
                "max_depth": max_depth,
                "intersected_voxel_idx": pts_idx
            }
        return ray_start, ray_dir, intersection_outputs, hits

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        return {
            'pos': samples['sampled_point_xyz'].requires_grad_(True),
            'ray': samples['sampled_point_ray_direction'],
            'dists': samples['sampled_point_distance']
        }

    @property
    def num_voxels(self):
        return self.vertices.size(0)

    def mesh_distance_field(self, inputs, outputs=['sigma']):
        query_points_l2 = (inputs['pos'] ** 2).sum(-1, keepdim=True)
        min_dis, min_idx = (self.sampled_points_l2.transpose(0, 1) + query_points_l2 - \
            2 * inputs['pos'] @ self.sampled_points.transpose(0, 1)).min(-1)
        inputs['sigma'] = (min_dis < self.gamma).type_as(min_dis) * self.sigma
        closed_sampled_points = F.embedding(min_idx, self.sampled_points)
        closed_sampled_normals = F.embedding(min_idx, self.sampled_normals)

        # get the projected points
        projected_dists = ((closed_sampled_points - inputs['pos']) * closed_sampled_normals).sum(-1, keepdim=True)
        projected_points = inputs['pos'] + projected_dists * closed_sampled_normals
        
        if self.sample_center and (self.values is not None):
            values = barycentric_interp(
                projected_points, 
                self.sampled_triangles[min_idx],
                F.embedding(F.embedding(min_idx, self.faces), self.values))
            inputs['emb'] = values
        else:
            values = projected_points

        inputs['mpos'] = torch.cat([values, projected_dists], -1)
        return inputs


@register_encoder('joint_volume_encoder')
class JointVolumeEncoder(VolumeEncoder):

    def __init__(self, args):
        Encoder.__init__(self, args)

        # load canonical shape
        self.mesh = o3d.io.read_triangle_mesh(self.args.mesh)
        if getattr(self.args, "new_tpose", None) is not None:  # this allows changing the canonical pose
            self.tpose_data = json.load(open(self.args.new_tpose))
        else:
            self.tpose_data = None

        # load texture information
        self.texture_encoder = None
        self.texture_to_deformation = getattr(self.args, "texture_to_deformation", False)
        self.no_textured_mesh = getattr(args, "no_textured_mesh", False)
        if getattr(self.args, "predict_texture_mask", False):
            self.texture_mask = nn.Parameter(torch.ones(512, 512) * 4, requires_grad=True)
        else:
            self.texture_mask = None
        if getattr(self.args, "texuv", None) is not None:  # this file saved uv coordinates of texture map
            
            def read_obj(filename):
                vt, ft = [], []
                for content in open(filename):
                    contents = content.strip().split(' ')
                    if contents[0] == 'vt':
                        vt.append([float(a) for a in contents[1:]])
                    if contents[0] == 'f':
                        ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
                return np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1
            
            vt, ft = read_obj(self.args.texuv)
            self.register_buffer("face_uv", torch.from_numpy(vt[ft]).float())   # num_faces x 3 x 2
            if getattr(self.args, "use_texture_encoder", False):
                from fairnr.modules.image_encoder import SpatialEncoder
                self.texture_encoder = SpatialEncoder(
                    num_layers=getattr(self.args, "texture_encoder_layers", 4)
                )
        else:
            self.face_uv = None

        # load and register buffers for joint weights of each node
        self.register_buffer("weights", torch.from_numpy(np.loadtxt(args.weights)).float())
        self.register_buffer("vertices", torch.from_numpy(np.array(self.mesh.vertices, dtype=np.float32)))
        self.register_buffer("faces", torch.from_numpy(np.array(self.mesh.triangles, dtype=np.long)))

        # setup hyperparameters
        self.num_joints              = self.weights.size(-1)
        self.model_dim               = getattr(args, "joint_feature_dim", 256)
        self.min_dis_eps             = getattr(args, "min_dis_eps", 0.3)  # minimum distance on the surface.
 
        self.filter_size             = args.joint_filter_size
        self.joint_filter            = NeRFPosEmbLinear(3, 6 * args.joint_filter_size, no_linear=True, cat_input=True, no_pi=True)
        
        self.use_global_rotation     = getattr(args, "use_global_rotation", False)
        self.use_local_coordinate    = getattr(args, "use_local_coordinate", False)
        
        self.disable_projection      = getattr(args, "disable_projection", False)    
        self.additional_deform       = getattr(args, "additional_deform", "pos")
        self.disable_transformer     = True
        self.disable_joint           = True
        self.disable_query_point     = getattr(args, "disable_query_point", False)
        self.disable_texture         = getattr(args, "disable_texture", False)
        self.deformation_layers      = getattr(args, "deformation_layers", 0)

        # setup input dimension
        input_dim                    = self.model_dim
        if self.disable_query_point:
            input_dim = 0
        if self.texture_to_deformation:
            if self.texture_encoder is not None:
                input_dim += self.texture_encoder.latent_size
            else:
                input_dim += 3

        # setup deformation
        if self.additional_deform == 'pos':
            self.skinning_deform = FCBlock(
                self.model_dim, 0, input_dim, 3, 
                outermost_linear=True, with_ln=False)
            self.skinning_deform.net[-1].weight.data *= 0
            self.skinning_deform.net[-1].bias.data *= 0

        pos_input_dim = 3 + 6 * args.joint_filter_size
        if not self.disable_query_point:
            self.merge_joint_field = FCBlock(
                self.model_dim, 2, pos_input_dim, 
                self.model_dim, outermost_linear=False, with_ln=False)            

        # other helpers
        from matplotlib import cm
        self.joint_colors = [cm.jet(k) for k in np.linspace(0,255,self.num_joints).astype('int')]
        self.joint_colors = torch.tensor([list(a) for a in self.joint_colors])
        self._runtime_caches = {}

        from fairnr.data.geometry import SkelModel
        self.skel_model = SkelModel()

    def get_edge(self, ray_start, ray_dir, *args, **kwargs):
        edge =  torch.zeros_like(ray_dir)
        edge[...,0] += 100./255.
        edge[...,1] += 200./255.
        edge[...,2] += 200./255.
        return edge

    @staticmethod
    def add_args(parser):
        parser.add_argument('--no-textured-mesh', action='store_true')
        parser.add_argument('--uv-textured-mesh', action='store_true')

        parser.add_argument('--mesh', type=str, help='path to the canonical mesh')
        parser.add_argument('--weights', type=str, help='path to the skinning weights')
        parser.add_argument('--texuv', type=str, default=None)
        parser.add_argument('--new-tpose', type=str, default=None)

        parser.add_argument('--use-texture-encoder', action='store_true', help='use an image encoder to extract features.')
        parser.add_argument('--texture-encoder-layers', type=int, default=4)
        parser.add_argument('--texture-to-deformation', action='store_true')
        parser.add_argument('--predict-texture-mask', action='store_true')
        parser.add_argument('--use-vertex-feature', action='store_true')
        parser.add_argument('--vertex-feature-gnn', action='store_true')
        parser.add_argument('--use-global-rotation', action='store_true')
        parser.add_argument('--use-local-coordinate', action='store_true')

        parser.add_argument('--min-dis-eps', type=float, default=0.15)
        parser.add_argument('--joint-input-motion', action='store_true', help='use three frames as inputs')
        parser.add_argument('--joint-feature-dim', type=int)
        parser.add_argument('--joint-embed-dim', type=int, default=64)
        parser.add_argument('--joint-filter-size', type=int, default=6)
        parser.add_argument('--joint_encoder_layers', type=int, default=2)
        parser.add_argument('--joint-decoder-layers', type=int, default=2)
        parser.add_argument('--deformation-layers', type=int, default=0)
        parser.add_argument('--joint-dropout', type=float, default=0.1)
        parser.add_argument('--joint-vertex-attention', action='store_true')
        parser.add_argument('--joint-back-projection', action='store_true')

        parser.add_argument('--disable-projection', action='store_true')
        parser.add_argument('--joint-query-unpose', action='store_true')
        parser.add_argument('--mixture-warp-space', action='store_true')
        parser.add_argument('--additional-deform', type=str, choices=['pos', 'weight'], default=None)

        parser.add_argument('--disable-joint', action='store_true')
        parser.add_argument('--disable-query-point', action='store_true')
        parser.add_argument('--disable-texture', action='store_true')
        parser.add_argument('--render-skeleton', action='store_true')

    def precompute(self, id=None, *args, **kwargs):
        # prepare the encoder before volume rendering
        assert id.size(0) == 1, 'only one shape per batch'    
        encoder_states = {
            "joints": kwargs["joints"],        # joint locations
            "joints_RT": kwargs["joints_RT"].permute(0,3,1,2).reshape(1,-1,16),  # joint RT
            "R": kwargs["rotation"],           # global rotation
            "T": kwargs["translation"],        # global translation
            "joints_pose": kwargs["pose"]      # joint feature
        }
        if self.tpose_data is not None:
            encoder_states['new_joints_RT'] = torch.tensor(self.tpose_data['joints_RT']).type_as(
                encoder_states['joints_RT']).permute(2,0,1).reshape(1,-1,16)
        
        if (self.face_uv is not None) and (not self.disable_texture):
            texture_map = kwargs['tex'].permute(0,3,1,2)  # 1 x [RGB] x 512 x 512
            if self.texture_mask is not None:
                texture_map = texture_map * torch.sigmoid(self.texture_mask[None, None, :,:])
            texture_rgb = torch.flip(texture_map, [2])    # texture rgb

            if self.texture_encoder is not None:
                texture_map = self.texture_encoder(texture_map)
            texture_map = torch.flip(texture_map, [2])

            encoder_states['texture_map'] = [texture_map]
            encoder_states['texture_rgb'] = [texture_rgb]

        # get the current mesh, mesh lays in normalized space.
        A = encoder_states["joints_RT"].squeeze(0)
        weighted_RT = (self.weights @ A).reshape(-1, 4, 4)   # N x 4 x 4
        vertices = torch.cat([self.vertices, self.vertices.new_ones(self.vertices.size(0), 1)], -1)  # N x 4 (normalized space)
        vertices = torch.einsum("ncd,nd->nc", weighted_RT, vertices)[:, :3]   #  (normalized space)
        encoder_states["vertices"] = vertices.unsqueeze(0)

        # apply inverse RT transform
        if not self.use_global_rotation:  # using the global rotation for the joints
            joints_normalized = torch.matmul(
                encoder_states["joints"][0] - encoder_states["T"][0], 
                encoder_states["R"][0].T)
        else:
            joints_normalized = encoder_states["joints"][0] - encoder_states["T"][0]
        encoder_states["joint_normalized"] = joints_normalized.unsqueeze(0)
        return encoder_states

    def set_num_updates(self, updates):
        self._updates = updates

    def compute_additional_loss(self, encoder_states):
        output = []
        if len(self._runtime_caches) > 0:
            for err_name in self._runtime_caches:
                if isinstance(self._runtime_caches[err_name], list):
                    err = torch.cat(self._runtime_caches[err_name], 0)
                else:
                    err = self._runtime_caches[err_name]
                output += [{"loss": err.mean(), "factor": 1.0, "name": err_name}]
        return output

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        S, V, P, _ = ray_dir.size()

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()

        center = encoder_states["joints"][0].mean(0, keepdim=True)
        radius = (encoder_states["joints"][0] - center).norm(2, -1).max()
        radius = radius * 1.88

        # we intersect the bounding sphere
        h_cet = center[None, :, :] - ray_start
        h_mid = (h_cet * ray_dir).sum(-1)
        h_dis = (h_cet ** 2).sum(-1) - h_mid ** 2
        hits = h_dis < (radius ** 2)

        # we intersect with mesh surface
        hit_ray_start = ray_start[hits]  # M x 3
        hit_ray_dir = ray_dir[hits]      # M x 3

        vertices_normalized = encoder_states["vertices"][0]   # normalized space
        vertices = torch.matmul(vertices_normalized, encoder_states["R"][0]) + encoder_states["T"][0]  # N x 3, mesh in posed/global space

        # -------------------- rendering mesh ----------------------------------- #
        mesh_depth, mesh_hits = h_mid.clone(), hits.clone()
        pts_idx, depth00, uv = triangle_ray_intersect(
            0, 0, 100, vertices.unsqueeze(0), self.faces.unsqueeze(0), 
            hit_ray_start.unsqueeze(0), hit_ray_dir.unsqueeze(0))
        hit_mesh = pts_idx.ne(-1).any(-1)
        depth000, index000 = depth00[:,:,:,0].masked_fill_(pts_idx.eq(-1), MAX_DEPTH).min(-1)
        mesh_depth[hits]   = depth000
        mesh_hits[hits]    = hit_mesh
        
        if (getattr(self, 'face_uv', None) is not None) and (not self.no_textured_mesh):
            select000 = pts_idx.gather(2, index000.unsqueeze(-1)).squeeze(-1)
            uv = uv.reshape(*uv.size()[:-1], -1, 2)
            u000, v000 = uv[..., 0], uv[..., 1]
            u000 = u000.gather(2, index000.unsqueeze(-1)).squeeze(-1)
            v000 = v000.gather(2, index000.unsqueeze(-1)).squeeze(-1)
            select000 = select000[hit_mesh]
            u000 = u000[hit_mesh]
            v000 = v000[hit_mesh]
            w000 = 1 - u000 - v000
            bary_coords = torch.stack([u000, v000, w000], 1)

            sampled_uvs = (self.face_uv[select000.long()] * bary_coords.unsqueeze(-1)).sum(1)
            if getattr(self.args, "uv_textured_mesh", False):
                sampled_tex = torch.cat([sampled_uvs, sampled_uvs.new_ones(sampled_uvs.size(0), 1)], 1)
                sampled_tex = sampled_tex[None,:,:]
            else:
                grids = sampled_uvs[None, :, None, :] * 2 - 1
                texture_rgb = encoder_states['texture_rgb'][0]
                textures = F.grid_sample(texture_rgb, grids, mode='bilinear', align_corners=False)
                sampled_tex = textures.permute(0, 2, 3, 1).reshape(1, -1, 3)
            
            mesh_color000 = mesh_depth.new_ones(*depth000.size(), 3)
            mesh_color000[hit_mesh] = sampled_tex
            
            mesh_color = mesh_depth.new_ones(*mesh_depth.size(), 3)
            mesh_color[hits] = mesh_color000

        else:

            mesh_color = self.get_edge(ray_start, ray_dir)

        # -------------------- rendering skeleton ----------------------------------- #
        if getattr(self.args, "rendering_skeleton", False):
            joints = encoder_states['joints'][0].cpu().numpy()
            skel_vertices = torch.from_numpy(self.skel_model(joints)).type_as(encoder_states['joints'])
            skel_faces = torch.from_numpy(self.skel_model.faces).type_as(self.faces)
            
            skel_depth, skel_hits = h_mid.clone(), hits.clone()
            pts_idx, depth00, uv = triangle_ray_intersect(
                0, 0, 100, skel_vertices.unsqueeze(0), skel_faces.unsqueeze(0), 
                hit_ray_start.unsqueeze(0), hit_ray_dir.unsqueeze(0))
            hit_mesh = pts_idx.ne(-1).any(-1)
            depth000, index000 = depth00[:,:,:,0].masked_fill_(pts_idx.eq(-1), MAX_DEPTH).min(-1)
            skel_depth[hits] = depth000
            skel_hits[hits] = hit_mesh
        else:
            skel_depth = None
            skel_hits = None
        
        # -------------------- ray-point cloud intersection  ----------------------------------- #
        from fairnr.clib import cloud_ray_intersect
        hit_mask, hit_min_depth, hit_max_depth = cloud_ray_intersect(self.min_dis_eps, vertices.unsqueeze(0), 
            hit_ray_start.unsqueeze(0), hit_ray_dir.unsqueeze(0))
        hit_mask, hit_min_depth, hit_max_depth = hit_mask[0,:,0].bool(), hit_min_depth[0,:,0], hit_max_depth[0,:,0]

        # min/max intersection
        min_depth, max_depth = h_mid.clone(), h_mid.clone()
        hits = hits.masked_scatter(hits, hit_mask)
        min_depth[hits] = hit_min_depth[hit_mask]
        max_depth[hits] = hit_max_depth[hit_mask]
        
        intersection_outputs = {
            "min_depth": min_depth.unsqueeze(-1),
            "max_depth": max_depth.unsqueeze(-1),
            "probs": ray_dir.new_ones(S, V * P, 1),
            "steps": ray_dir.new_ones(S, V * P) * self.args.fixed_num_samples,
            "intersected_voxel_idx": ray_dir.new_zeros(S, V * P, 1).int(),
            "mesh_hits": mesh_hits,
            "mesh_depth": mesh_depth,
            "mesh_color": mesh_color, 
        }
        if skel_depth is not None:
            intersection_outputs["skel_depth"] = skel_depth
            intersection_outputs["skel_hits"] = skel_hits
        return ray_start, ray_dir, intersection_outputs, hits

    def forward(self, samples, encoder_states=None):
        sampled_dir = samples.get('sampled_point_ray_direction', None)
        sampled_dis = samples.get('sampled_point_distance', None)

        # prepare inputs for implicit field
        inputs = {
            'ray': sampled_dir, 
            'dists': sampled_dis}

        if encoder_states is not None:  # attention with joints
            if 'sampled_normalized_xyz' not in samples:   # sampled points in normalized space
                sampled_normalized = torch.matmul(
                    samples['sampled_point_xyz'] - encoder_states["T"], 
                    encoder_states["R"].T)
                if self.use_global_rotation:   # sampled points without global rotation
                    sampled_global_rotation = samples['sampled_point_xyz'] - encoder_states["T"]
            else:
                assert (not self.use_global_rotation), "not support"
                sampled_normalized = samples['sampled_normalized_xyz']

            # query from the nearest vertex/face for features/weights
            vertices = encoder_states['vertices'].contiguous()   # mesh in normalized space
           
            from fairnr.clib._ext import point_face_dist_forward
            triangles = F.embedding(self.faces, vertices)
            l_idx = torch.tensor([0,]).type_as(self.faces)
            min_dis, min_face_idx, w0, w1, w2 = point_face_dist_forward(
                sampled_normalized, l_idx, triangles, l_idx, sampled_normalized.size(0)
            )
            bary_coords = torch.stack([w0, w1, w2], 1)   # B x 3
            sampled_uvs = (self.face_uv[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)

            if self.use_local_coordinate:
                # compute triangle normals
                A, B, C = triangles[:,0], triangles[:,1], triangles[:,2]
                triangle_normals = torch.cross((B - A), (C - B), 1)
                faces_xyzs = (triangles[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)
                insideout = ((sampled_normalized - faces_xyzs) * triangle_normals[min_face_idx]).sum(-1).sign()
                local_coords = torch.cat([sampled_uvs * 2 - 1, (min_dis.sqrt() * insideout)[:, None]], 1)
            
            if (self.face_uv is not None) and (not self.disable_texture):
                texture_map = encoder_states['texture_map'][0]  # query texture map here.
                f_dim = texture_map.size(1)
                grids = sampled_uvs[None, :, None, :] * 2 - 1     # 1 x num_samples x 1 x 2
                textures = F.grid_sample(texture_map, grids, mode='bilinear', align_corners=False)  # 1 x [Feat] x num_samples x 1
                sampled_tex = textures.permute(0, 2, 3, 1).reshape(-1, f_dim)  # (num_samples) x [RGB/feature]
                inputs['texture'] = sampled_tex
                       
            if self.use_local_coordinate:
                x = self.joint_filter(local_coords)       
            elif not self.use_global_rotation:
                x = self.joint_filter(sampled_normalized)
            else:
                x = self.joint_filter(sampled_global_rotation)

            if (not self.disable_joint) or self.disable_query_point:
                joints_pose = encoder_states['joints_pose']
                x = torch.cat([x, joints_pose.expand(x.size(0), -1)], -1)
            if not self.disable_query_point:
                attnout = self.merge_joint_field(x)
            inputs["attnout"] = attnout

            if not self.disable_projection:   # project to canonical space?
                face_weights = F.embedding(self.faces, self.weights)  # num_face x 3 x 24
                weights = (face_weights[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1) # num_samples x 24
               
                # use weights to apply joint-transformation
                inputs["weights"] = weights
                inputs["joint_colors"] = weights @ self.joint_colors.type_as(weights)
                inputs["posed_pos"] = sampled_normalized    # sampled points in the normalized space, not global space

                A = encoder_states["joints_RT"]
                weighted_RT = (weights @ A).reshape(-1, 4, 4)   # N x 4 x 4
                sampled_normalized = torch.cat([sampled_normalized, sampled_normalized.new_ones(sampled_normalized.size(0), 1)], -1)  # N x 4
                sampled_canonical = torch.einsum("ncd,nd->nc", torch.inverse(weighted_RT), sampled_normalized)
                
                if 'offsets' in encoder_states:   # transform space
                    nn_offsets = (F.embedding(self.faces, encoder_states["offsets"])[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1) 
                    sampled_canonical[:, :3] = sampled_canonical[:, :3] + nn_offsets
                    
                if self.tpose_data is not None:
                    new_weighted_RT = (weights @ encoder_states['new_joints_RT']).reshape(-1, 4, 4)   # N x 4 x 4
                    sampled_canonical = torch.einsum("ncd,nd->nc", new_weighted_RT, sampled_canonical)
                sampled_canonical = sampled_canonical[:,:3]
    
                if self.additional_deform == "pos":
                    if not self.texture_to_deformation:
                        deform_input = attnout
                    else:
                        assert self.face_uv is not None
                        assert (not self.disable_texture)
                        if not self.disable_query_point:
                            deform_input = torch.cat([attnout, sampled_tex], -1)
                        else:
                            deform_input = sampled_tex
                    
                    sampled_canonical = sampled_canonical + self.skinning_deform(deform_input)
                inputs["pos"] = sampled_canonical
            
            else:
                samples = sampled_normalized if not self.use_global_rotation else sampled_global_rotation
                if self.additional_deform == "pos":
                    if not self.texture_to_deformation:
                        deform_input = attnout
                    
                    else:
                        assert self.face_uv is not None
                        assert (not self.disable_texture)
                        
                        if not self.disable_query_point:
                            deform_input = torch.cat([attnout, sampled_tex], -1)
                        else:
                            deform_input = sampled_tex      
                    samples = samples + self.skinning_deform(deform_input)
                
                inputs["pos"] = samples
                
        return inputs

    def postcompute(self, inputs, encoder_states):
        pass

def bbox2voxels(bbox, voxel_size):
    vox_min, vox_max = bbox[:3], bbox[3:]
    steps = ((vox_max - vox_min) / voxel_size).round().astype('int64') + 1
    x, y, z = [c.reshape(-1).astype('float32') for c in np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]))]
    x, y, z = x * voxel_size + vox_min[0], y * voxel_size + vox_min[1], z * voxel_size + vox_min[2]
    return np.stack([x, y, z]).T.astype('float32')

