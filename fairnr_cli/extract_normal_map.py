#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This code is used for extact voxels/meshes from the learne model
"""
import logging
import numpy as np
import torch
import sys, os
import cv2
import argparse
import open3d as o3d

from fairseq import options
from fairseq import checkpoint_utils
from fairnr.data.data_utils import load_texcoords
from fairnr.tools.tex.iso import IsoColoredRenderer
from opendr.geometry import VertNormals


def main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger('fairnr_cli.extract')
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    
    # resolution = args.tex_resolution
    # mesh = o3d.io.read_triangle_mesh(args.mesh)
    # v, f = np.array(mesh.vertices), np.array(mesh.triangles)
    # vt, ft = load_texcoords(args.texuv)
    # weight = np.loadtxt(args.weights)

    # iso_vis = IsoColoredRenderer(vt, ft, f, resolution)
    # vn = VertNormals(f=f, v=v)
    # normal_tex = iso_vis.render(vn / 2.0 + 0.5)
    # cv2.imwrite('results/debug_normal12.png', normal_tex * 255) 
    # from fairseq import pdb;pdb.set_trace()

    resolution = 512

    model_file = '/private/home/jgu/data/shapenet/DEBUG/full_data/testing/smpl_uv.obj'
    pose_file = '/private/home/jgu/data/shapenet/DEBUG/full_data/testing/canonical.obj'

    vt, ft = load_texcoords(model_file)

    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(pose_file)
    v, f = np.array(mesh.vertices), np.array(mesh.triangles)
    # v, f = read_obj_track(pose_file)
    # f -= 1

    # iso = Isomapper(vt, ft, f, resolution, bgcolor=bgcolor)
    iso_vis = IsoColoredRenderer(vt, ft, f, resolution)

    vn = VertNormals(f=f, v=v)
    normal_tex = iso_vis.render(vn / 2.0 + 0.5)
    cv2.imwrite('results/debug_outpu24t.png', normal_tex*255) 


def cli_main():
    parser = argparse.ArgumentParser(description='Extract geometry from a trained model (only for learnable embeddings).')
    parser.add_argument('--mesh', type=str, help='path to the canonical mesh')
    parser.add_argument('--weights', type=str, help='path to the skinning weights')
    parser.add_argument('--texuv', type=str, default=None)
    parser.add_argument('--joint-data', type=str)
    parser.add_argument('--tex-resolution', type=int, default=512)
    parser.add_argument('--user-dir', default='fairnr')
    parser.add_argument('--cpu', action='store_true')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
