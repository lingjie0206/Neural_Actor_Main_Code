#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cPickle as pkl

"""

This script creates a .pkl file using the given camera intrinsics.

Example:
$ python camera.pkl 1080 1080 -f 900.0 900.0
$ python create_camera.py camera.pkl 960 540 -f 7.613159331190774e+02 7.605554555493511e+02 -c 4.853989013687801e+02 2.806341319344401e+02
$ python create_camera.py camera.pkl 960 540 -f 780.0 780.0 -t -2.3474597930908203 3913.475830078125 -844.2236328125 -rt 1.6280 0.0004 -0.0004
"""

parser = argparse.ArgumentParser()
parser.add_argument('out', type=str, help="Output file (.pkl)")
parser.add_argument('width', type=int, help="Frame width in px")
parser.add_argument('height', type=int, help="Frame height in px")
parser.add_argument('-f', type=float, nargs='*', help="Focal length in px (2,)")
parser.add_argument('-c', type=float, nargs='*', help="Principal point in px (2,)")
parser.add_argument('-k', type=float, nargs='*', help="Distortion coefficients (5,)")
parser.add_argument('-t', type=float, nargs='*', help="extrinsic:translation vector (3,)")
parser.add_argument('-rt', type=float, nargs='*', help="extrinsic:rotation vector (3,)")


args = parser.parse_args()

camera_data = {
    'camera_t': np.zeros(3),
    'camera_rt': np.zeros(3),
    'camera_f': np.array([args.width, args.width]),
    'camera_c': np.array([args.width, args.height]) / 2.,
    'camera_k': np.zeros(5),
    'width': args.width,
    'height': args.height,
}

if args.f is not None:
    if len(args.f) is not 2:
        raise Exception('Focal length should be of shape (2,)')

    camera_data['camera_f'] = np.array(args.f)

if args.c is not None:
    if len(args.c) is not 2:
        raise Exception('Principal point should be of shape (2,)')

    camera_data['camera_c'] = np.array(args.c)

if args.k is not None:
    if len(args.k) is not 5:
        raise Exception('Distortion coefficients should be of shape (5,)')

    camera_data['camera_k'] = np.array(args.k)
	
if args.t is not None:
    if len(args.t) is not 3:
        raise Exception('Extrinsic translation should be of shape (3,)')

    camera_data['camera_t'] = np.array(args.t)
	
if args.rt is not None:
    if len(args.rt) is not 3:
        raise Exception('Extrinsic rotation vector should be of shape (3,)')

    camera_data['camera_rt'] = np.array(args.rt)

with open(args.out, 'wb') as f:
    pkl.dump(camera_data, f, protocol=2)
