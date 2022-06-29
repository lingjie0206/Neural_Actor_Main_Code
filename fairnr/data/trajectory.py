# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

TRAJECTORY_REGISTRY = {}


def register_traj(name):
    def register_traj_fn(fn):
        if name in TRAJECTORY_REGISTRY:
            raise ValueError('Cannot register duplicate trajectory ({})'.format(name))
        TRAJECTORY_REGISTRY[name] = fn
        return fn
    return register_traj_fn


def get_trajectory(name):
    return TRAJECTORY_REGISTRY.get(name, None)


@register_traj('circle')
def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1, offset=None):
    if offset is not None:
        x0, y0, z0 = eval(offset)
    else:
        x0, y0, z0 = 0, 0, 0
    
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t+t0) + x0, radius * np.sin(r * t+t0) + y0, h + z0]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t+t0) + x0, h + y0, radius * np.sin(r * t+t0) + z0]
    else:
        return lambda t: [h + x0, radius * np.cos(r * t+t0) + y0, radius * np.sin(r * t+t0) + z0]


@register_traj('zoomin_circle')
def zoomin_circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    ra = lambda t: 0.1 + abs(4.0 - t * 2 / np.pi)

    if axis == 'z':
        return lambda t: [radius * ra(t) * np.cos(r * t+t0), radius * ra(t) * np.sin(r * t+t0), h]
    elif axis == 'y':
        return lambda t: [radius * ra(t) * np.cos(r * t+t0), h, radius * ra(t) * np.sin(r * t+t0)]
    else:
        return lambda t: [h, radius * (4.2 - t * 2 / np.pi) * np.cos(r * t+t0), radius * (4.2 - t * 2 / np.pi) * np.sin(r * t+t0)]


@register_traj('zoomin_line')
def zoomin_line(radius=3.5, h=0.0, axis='z', t0=0, r=1, min_r=0.0001, max_r=10, step_r=10):
    ra = lambda t: min_r + (max_r - min_r) * t * 180 / np.pi / step_r

    if axis == 'z':
        return lambda t: [radius * ra(t) * np.cos(t0), radius * ra(t) * np.sin(t0), h * ra(t)]
    elif axis == 'y':
        return lambda t: [radius * ra(t) * np.cos(t0), h, radius * ra(t) * np.sin(t0)]
    else:
        return lambda t: [h, radius * (4.2 - t * 2 / np.pi) * np.cos(r * t+t0), radius * (4.2 - t * 2 / np.pi) * np.sin(r * t+t0)]
