#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import h5py
import argparse
import numpy as np
import json
import glob
import open3d as o3d

try:
    import cPickle as pkl
except Exception:
    import pickle as pkl
    
import os

from opendr.geometry import VertNormals
from tex.iso import IsoColoredRenderer
from tqdm import tqdm
from util.logger import log
from multiprocessing import Pool


def load_texcoords(filename):
    vt, ft = [], []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'vt':
            vt.append([float(a) for a in contents[1:]])
        if contents[0] == 'f':
            ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
    return np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1


def main(args, id=0, total=1):
    resolution = args.tex_resolution
    vt, ft = load_texcoords(args.texuv)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    v, f = np.array(mesh.vertices), np.array(mesh.triangles)
    v_ext = np.concatenate([v, np.ones((v.shape[0], 1))], axis=-1)[:,:,None]
    weights = np.loadtxt(args.weights)
    iso_vis = IsoColoredRenderer(vt, ft, f, resolution)

    generator = tqdm(sorted(glob.glob(args.joint_data + '/*.json'))) if id == 0 \
        else sorted(glob.glob(args.joint_data + '/*.json'))
    
    s = 0
    for i, joint_file in enumerate(generator):
        if i % total != id:
            continue
        jname = os.path.splitext(os.path.basename(joint_file))[0]
        jdata = json.load(open(joint_file))
        R = np.asarray(jdata['rotation'])
        T = np.asarray(jdata['translation'])
        A = np.asarray(jdata['joints_RT'])
        G = (weights @ A.transpose(2,0,1).reshape(-1,16)).reshape(-1, 4, 4)
        v_normalized = np.matmul(G, v_ext)[:, :3, 0]
        v_posed = np.matmul(v_normalized, R) + T
        vn = VertNormals(f=f, v=v_posed)

        normal_tex = iso_vis.render(vn / 2.0 + 0.5)
        cv2.imwrite('{}/{}.png'.format(args.output_path, jname), normal_tex*255)
        s += 1
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract geometry from a trained model (only for learnable embeddings).')
    parser.add_argument('--mesh', type=str, help='path to the canonical mesh')
    parser.add_argument('--weights', type=str, help='path to the skinning weights')
    parser.add_argument('--texuv', type=str, default=None)
    parser.add_argument('--joint-data', type=str)
    parser.add_argument('--tex-resolution', type=int, default=512)
    parser.add_argument('--output-path', type=str, help='path to save normal maps')
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)

    result_objs = []
    # main(args, 0, 1)
    with Pool(20) as p:
        for i in range(20):
            result_objs.append(p.apply_async(main, [args, i, 20]))
        results = [result.get() for result in result_objs]
    log.info("Done")