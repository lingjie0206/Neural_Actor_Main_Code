# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import random, os, glob

from fairnr.data.geometry import get_ray_direction, r6d2mat

torch.autograd.set_detect_anomaly(True)
TINY = 1e-9
READER_REGISTRY = {}

def register_reader(name):
    def register_reader_cls(cls):
        if name in READER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        READER_REGISTRY[name] = cls
        return cls
    return register_reader_cls


def get_reader(name):
    if name not in READER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return READER_REGISTRY[name]


@register_reader('abstract_reader')
class Reader(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass


@register_reader('image_reader')
class ImageReader(Reader):
    """
    basic image reader
    """
    def __init__(self, args):
        super().__init__(args)
        self.num_pixels = args.pixel_per_view
        self.no_sampling = getattr(args, "no_sampling_at_reader", False)
        self.deltas = None
        self.expos = None
        self.all_data = self.find_data()
        if getattr(args, "trainable_extrinsics", False):
            self.all_data_idx = {data_img: (s, v) 
                for s, data in enumerate(self.all_data) 
                for v, data_img in enumerate(data)}
            self.deltas = nn.ParameterList([
                nn.Parameter(torch.tensor(
                    [[1., 0., 0., 0., 1., 0., 0., 0., 0.]]).repeat(len(data), 1))
                for data in self.all_data])
        if getattr(args, "trainable_expo", False):
            self.all_data_idx = {data_img: (s, v) 
                for s, data in enumerate(self.all_data) 
                for v, data_img in enumerate(data)}
            self.expos = nn.ParameterList([
                nn.Parameter(torch.tensor(
                    [[0.5, 0.]]).repeat(len(data), 1))
                for data in self.all_data])

    def find_data(self):
        paths = self.args.data
        try:
            if os.path.isdir(paths):
                self.paths = [paths]
            else:
                self.paths = [line.strip() for line in open(paths)]
        except Exception:
            return None
        return [sorted(glob.glob("{}/rgb/*".format(p))) for p in self.paths]
        
    @staticmethod
    def add_args(parser):
        parser.add_argument('--pixel-per-view', type=float, metavar='N', 
                            help='number of pixels sampled for each view')
        parser.add_argument("--uniform-sampling", action='store_true',
                            help="uniformly sample pixels no matter with masks or not.")
        parser.add_argument("--sampling-on-mask", nargs='?', const=0.9, type=float,
                            help="this value determined the probability of sampling rays on masks")
        parser.add_argument("--sampling-at-center", type=float,
                            help="only useful for training where we restrict sampling at center of the image")
        parser.add_argument("--sampling-on-bbox", action='store_true',
                            help="sampling points to close to the mask")
        parser.add_argument("--sampling-patch-size", type=int, 
                            help="sample pixels based on patches instead of independent pixels")
        parser.add_argument("--sampling-skipping-size", type=int,
                            help="sample pixels if we have skipped pixels")
        parser.add_argument("--no-sampling-at-reader", action='store_true',
                            help="do not perform sampling.")
        parser.add_argument("--trainable-extrinsics", action='store_true',
                            help="if set, we assume extrinsics are trainable. We use 6D representations for rotation")
        parser.add_argument("--trainable-expo", action='store_true')
        
    def forward(self, uv, intrinsics, extrinsics, size, path=None, **kwargs):
        S, V = uv.size()[:2]
        if (not self.training) or self.no_sampling:
            uv = uv.reshape(S, V, 2, -1, 1, 1)
            flatten_uv = uv.reshape(S, V, 2, -1)
        else:
            uv, _ = self.sample_pixels(uv, size, **kwargs)
            flatten_uv = uv.reshape(S, V, 2, -1)

        # go over all shapes
        ray_start, ray_dir = [[] for _ in range(S)], [[] for _ in range(S)]
        curr_expos = [[] for _ in range(S)]
        for s in range(S):
            for v in range(V):
                ixt = intrinsics[s] if intrinsics.dim() == 3 else intrinsics[s, v]
                ext = extrinsics[s, v]
                translation, rotation = ext[:3, 3], ext[:3, :3]
                if (self.expos is not None) and (path is not None):
                    shape_id, view_id = self.all_data_idx[path[s][v]]
                    curr_expos[s] += [self.expos[shape_id][view_id]]
                if (self.deltas is not None) and (path is not None):
                    shape_id, view_id = self.all_data_idx[path[s][v]]
                    delta = self.deltas[shape_id][view_id]
                    d_t, d_r = delta[6:], r6d2mat(delta[None, :6]).squeeze(0)
                    rotation = rotation @ d_r
                    translation = translation + d_t
                    ext = torch.cat([torch.cat([rotation, translation[:, None]], 1), ext[3:]], 0)
                ray_start[s] += [translation]
                ray_dir[s] += [get_ray_direction(translation, flatten_uv[s, v], ixt, ext, 1)]
        if len(curr_expos[0]) > 0:
            self.curr_expos = torch.stack([torch.stack([b for b in a], 0) for a in curr_expos], 0)
        else:
            self.curr_expos = None        
        ray_start = torch.stack([torch.stack(r) for r in ray_start])
        ray_dir = torch.stack([torch.stack(r) for r in ray_dir])
        return ray_start.unsqueeze(-2), ray_dir.transpose(2, 3), uv
    
    @torch.no_grad()
    def sample_pixels(self, uv, size, alpha=None, mask=None, **kwargs):
        H, W = int(size[0,0,0]), int(size[0,0,1])
        S, V = uv.size()[:2]

        if getattr(self.args, "uniform_sampling", False):
            mask = uv.new_ones(S, V, H, W).float()
            probs = mask / (mask.sum() + 1e-8)
        
        else:
            if mask is None:
                if alpha is not None:
                    mask = (alpha > 0)
                else:
                    mask = uv.new_ones(S, V, uv.size(-1)).bool()
            mask = mask.float().reshape(S, V, H, W)

            if self.args.sampling_at_center < 1.0:
                r = (1 - self.args.sampling_at_center) / 2.0
                mask0 = mask.new_zeros(S, V, H, W)
                mask0[:, :, int(H * r): H - int(H * r), int(W * r): W - int(W * r)] = 1
                mask = mask * mask0
            
            if self.args.sampling_on_bbox:
                x_has_points = mask.sum(2, keepdim=True) > 0
                y_has_points = mask.sum(3, keepdim=True) > 0
                mask = (x_has_points & y_has_points).float()  

            probs = mask / (mask.sum() + 1e-8)
            if self.args.sampling_on_mask > 0.0:
                probs = self.args.sampling_on_mask * probs + (1 - self.args.sampling_on_mask) * 1.0 / (H * W)

        num_pixels = int(self.args.pixel_per_view)
        patch_size, skip_size = self.args.sampling_patch_size, self.args.sampling_skipping_size
        C = patch_size * skip_size
        
        if C > 1:
            H2 = (H // C + 1) * C if (H // C) * C < H else H
            W2 = (W // C + 1) * C if (W // C) * C < W else W
            probs_full = probs.new_zeros(S, V, H2, W2)
            probs_full[:,:,:H,:W] = probs
            probs_full = probs_full.reshape(S, V, H2 // C, C, W2 // C, C).sum(3).sum(-1)
            if H2 > H:
                probs_full[:, :, -1, :] *= 0
            if W2 > W:
                probs_full[:, :, :, -1] *= 0
            num_pixels = num_pixels // patch_size // patch_size
        else:
            H2, W2, probs_full = H, W, probs  # for compitibility

        flatten_probs = probs_full.reshape(S, V, -1) 
        sampled_index = sampling_without_replacement(torch.log(flatten_probs+ TINY), num_pixels)
        sampled_masks = torch.zeros_like(flatten_probs).scatter_(-1, sampled_index, 1).reshape(S, V, H2 // C, W2 // C)
        
        if C > 1:
            sampled_masks = sampled_masks[:, :, :, None, :, None].repeat(
                1, 1, 1, patch_size, 1, patch_size).reshape(S, V, H2 // skip_size, W2 // skip_size)
            
            if skip_size > 1:
                full_datamask = sampled_masks.new_zeros(S, V, skip_size * skip_size, H2 // skip_size, W2 // skip_size)
                full_index = torch.randint(skip_size*skip_size, (S, V))
                for i in range(S):
                    for j in range(V):
                        full_datamask[i, j, full_index[i, j]] = sampled_masks[i, j]
                sampled_masks = full_datamask.reshape(
                    S, V, skip_size, skip_size, H2 // skip_size, W2 // skip_size).permute(0, 1, 4, 2, 5, 3).reshape(S, V, H2, W2)
                # import imageio
                # imageio.imsave("results/example.png", sampled_masks[0,0].cpu().numpy())
            
            sampled_masks = sampled_masks[:,:,:H,:W]

        # sampled_masks = sampled_masks * mask
        X, Y = uv[:,:,0].reshape(S, V, H, W), uv[:,:,1].reshape(S, V, H, W)
        X = X[sampled_masks>0].reshape(S, V, 1, -1, patch_size, patch_size)
        Y = Y[sampled_masks>0].reshape(S, V, 1, -1, patch_size, patch_size)
        return torch.cat([X, Y], 2), sampled_masks


def sampling_without_replacement(logp, k):
    def gumbel_like(u):
        return -torch.log(-torch.log(torch.rand_like(u) + TINY) + TINY)
    scores = logp + gumbel_like(logp)
    return scores.topk(k, dim=-1)[1]