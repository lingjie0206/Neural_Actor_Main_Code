# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
import copy
import numpy as np
import cv2
import torch
import logging
import json

from collections import defaultdict
from fairseq.data import FairseqDataset, BaseWrapperDataset
from . import data_utils, geometry, trajectory


logger = logging.getLogger(__name__)


class ShapeDataset(FairseqDataset):
    """
    A dataset that only returns data per shape
    """
    def __init__(self, 
                paths, 
                preload=True,
                repeat=1,
                subsample_valid=-1,
                start_end=None,
                load_vertex=False,
                ids=None):

        self.paths = self.find_paths(paths)
        self.subsample_valid = subsample_valid
        self.start_end = start_end
        self.total_num_shape = len(self.paths)
        self.cache = None
        self.repeat = repeat
        
        # -- load per-shape data
        _data_per_shape = {}
        _data_per_shape['shape'] = list(range(len(self.paths)))
        _ixts = self.find_intrinsics()
        if len(_ixts) > 0:
            _data_per_shape['ixt'] = _ixts
        _glbs = self.find_global()
        if len(_glbs) > 0:
            _data_per_shape['glb'] = _glbs
        _jnts = self.find_joints()
        if len(_jnts) > 0:
            _data_per_shape['jnt'] = _jnts
        _texs = self.find_textures()
        if len(_texs) > 0:
            _data_per_shape['tex'] = _texs

        if load_vertex:
            _vexs = self.find_vertex(vertex_fname="vertex.txt")
            if len(_vexs) > 0:
                _data_per_shape['vertex'] = _vexs
        
        if self.start_end is not None:
            
            start, end = eval(self.start_end)
            for key in _data_per_shape:
                _data_per_shape[key] = _data_per_shape[key][start: end]
            self.paths = self.paths[start: end]
            self.total_num_shape = len(self.paths)
            
        if self.subsample_valid > -1:
            for key in _data_per_shape:
                _data_per_shape[key] = _data_per_shape[key][::self.subsample_valid]
            self.paths = self.paths[::self.subsample_valid]
            self.total_num_shape = len(self.paths)
        
        print('real {} frames'.format(self.total_num_shape))

        # from fairseq import pdb;pdb.set_trace()
        # group the data..
        data_list = []
        for r in range(repeat):
            # HACK: making several copies to enable multi-GPU usage.
            if r == 0 and preload:
                self.cache = []
                logger.info('pre-load the dataset into memory.')

            for id in range(self.total_num_shape): 
                element = {}
                for key in _data_per_shape:
                    try:
                        element[key] = _data_per_shape[key][id]
                    except IndexError as e:
                        print(key, id)
                        raise e

                data_list.append(element)

                if r == 0 and preload:
                    self.cache += [self._load_batch(data_list, id)]

        # group the data together
        self.data = data_list

    def find_paths(self, paths):
        if os.path.isdir(paths):
            paths = [paths]
        else:
            paths = [line.strip() for line in open(paths)]
        return paths

    def find_intrinsics(self):
        ixt_list = []
        for path in self.paths:
            if os.path.exists(path + '/intrinsic.txt'):
                ixt_list.append(path + '/intrinsic.txt')
            elif os.path.exists(path + '/intrinsics.txt'):
                ixt_list.append(path + '/intrinsics.txt')
        return ixt_list

    def find_global(self):
        glb_list = []
        for path in self.paths:
            if os.path.exists(path + '/global.txt'):
                glb_list.append(path + '/global.txt')
        return glb_list

    def find_vertex(self, vertex_fname="vertex.txt"):
        vex_list = []
        for path in self.paths:
            if os.path.exists(os.path.join(path, vertex_fname)):
                vex_list.append(os.path.join(path, vertex_fname))
        return vex_list

    def find_joints(self, joint_fname="joints.json"):
        jst_list = []
        for path in self.paths:
            if os.path.exists(os.path.join(path, joint_fname)):
                jst_list.append(os.path.join(path, joint_fname))
        return jst_list

    def find_textures(self):
        return []   # template

    def _load_shape(self, packed_data):  
        intrinsics = data_utils.load_intrinsics(packed_data['ixt']).astype('float32') \
            if packed_data.get('ixt', None) is not None else None
        shape_id = packed_data['shape']
        shape_data = {'intrinsics': intrinsics, 'id': shape_id}
        if packed_data.get('glb', None) is not None:   # additional global feature (if any)
            shape_data['global_index'] = np.loadtxt(packed_data['glb']).astype('int64')
        if packed_data.get('vertex', None) is not None:
            shape_data['vertex'] = np.loadtxt(packed_data.get('vertex', None)).astype('float32')[:, :3]
        if packed_data.get('jnt', None) is not None:
            data = json.load(open(packed_data['jnt']))
            for key in data:
                shape_data[key] = np.asarray(data[key]).astype('float32')
            
            rodrigues = lambda x : cv2.Rodrigues(x)[0]
   
            Rh = rodrigues(shape_data['rotation'].transpose(1,0))[:,0]
            if len(shape_data['pose'].shape) == 2:
                shape_data['pose'] = np.concatenate([Rh[None, :], shape_data['pose'][:, 3:]], -1)
            else:
                shape_data['pose'] = np.concatenate([Rh, shape_data['pose'][3:]], -1)
            
        if packed_data.get('tex', None) is not None:
            shape_data['tex'] = data_utils.load_rgb(packed_data['tex'], with_alpha=False)
        return shape_data

    def _load_batch(self, data, index):
        return index, self._load_shape(data[index])

    def __getitem__(self, index):
        if self.cache is not None:
            return self.cache[index % self.total_num_shape][0], \
                   self.cache[index % self.total_num_shape][1]
        return self._load_batch(self.data, index)

    def __len__(self):
        return len(self.data)

    def num_tokens(self, index):
        return 1

    def _collater(self, samples):
        results = {}

        results['shape'] = torch.from_numpy(np.array([s[0] for s in samples]))    
        for key in samples[0][1]:
            if samples[0][1][key] is not None:
                results[key] = torch.from_numpy(
                    np.array([s[1][key] for s in samples]))
            else:
                results[key] = None
        return results

    def collater(self, samples):
        try:
            results = self._collater(samples)
        except IndexError:
            results = None
        return results


class ShapeViewDataset(ShapeDataset):
    """
    A dataset contains a series of images renderred offline for an object.
    """

    def __init__(self, 
                paths, 
                views,
                num_view,
                subsample_valid=-1,
                start_end=None,
                resolution=None, 
                load_depth=False,
                load_mask=False,
                train=True,
                preload=True,
                repeat=1,
                binarize=True,
                bg_color="1,1,1",
                min_color=-1,
                ids=None,
                load_all_images=False,
                load_rgb_name="rgb"):
        super().__init__(paths, False, repeat, subsample_valid, start_end, ids)

        self.train = train
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.load_all_images = load_all_images
        self.views = views
        self.num_view = num_view

        if isinstance(resolution, str):
            self.resolution = [int(r) for r in resolution.split('x')]
        else:
            self.resolution = [resolution, resolution]
        self.world2camera = True
        self.cache_view = None
        
        bg_color = [float(b) for b in bg_color.split(',')] \
            if isinstance(bg_color, str) else [bg_color]
        if min_color == -1:
            bg_color = [b * 2 - 1 for b in bg_color]
        if len(bg_color) == 1:
            bg_color = bg_color + bg_color + bg_color
        self.bg_color = bg_color
        self.min_color = min_color
        self.apply_mask_color = (self.bg_color[0] >= -1) & (self.bg_color[0] <= 1)  # if need to apply

        # -- load per-view data
        self.no_rgb = False
        _data_per_view = {}
        _data_per_view['rgb'] = self.find_rgb(load_rgb_name)
        
        if _data_per_view['rgb'] is not None:
            _data_per_view['ext'] = self.find_extrinsics()
            if self.find_intrinsics_per_view() is not None:
                _data_per_view['ixt_v'] = self.find_intrinsics_per_view()
            if self.load_depth:
                _data_per_view['dep'] = self.find_depth()
            if self.load_mask:
                _data_per_view['mask'] = self.find_mask()
            _data_per_view['view'] = self.summary_view_data(_data_per_view)
            
            # group the data.
            _index = 0
            for r in range(repeat):
                # HACK: making several copies to enable multi-GPU usage.
                if r == 0 and preload:
                    self.cache = []
                    logger.info('pre-load the dataset into memory.')

                for id in range(self.total_num_shape): 
                    element = {}
                    total_num_view = len(_data_per_view['rgb'][id])
                    perm_ids = np.random.permutation(total_num_view) if train else np.arange(total_num_view)
                    for key in _data_per_view:
                        element[key] = [_data_per_view[key][id][i] for i in perm_ids]
                    self.data[_index].update(element)

                    if r == 0 and preload:
                        phase_name = f"{'train' if self.train else 'valid'}" + \
                                    f".{self.resolution[0]}x{self.resolution[1]}" + \
                                    f"{'.d' if load_depth else ''}" + \
                                    f"{'.m' if load_mask else ''}" + \
                                    f"{'b' if not self.apply_mask_color else ''}" + \
                                    "_full"
                        logger.info("preload {}-{}".format(id, phase_name))
                        if binarize:
                            cache = self._load_binary(id, np.arange(total_num_view), phase_name)
                        else:
                            cache = self._load_batch(self.data, id, np.arange(total_num_view))
                        self.cache += [cache]
                    _index += 1

            # group the data together
            self.data_index = []
            for i, d in enumerate(self.data):
                if self.train:
                    index_list = list(range(len(d['rgb'])))
                    self.data_index.append(
                        data_utils.InfIndex(index_list, shuffle=True)
                    )
                else:
                    copy_id = i // self.total_num_shape
                    index_list = []
                    for j in range(copy_id * num_view, copy_id * num_view + num_view):
                        index_list.append(j % len(d['rgb']))
                    self.data_index.append(
                        data_utils.InfIndex(index_list, shuffle=False)
                    )

        else:
            self.no_rgb = True

    def _load_binary(self, id, views, phase='train'):
        root = os.path.dirname(self.data[id]['shape'])
        npzfile = os.path.join(root, '{}.npz'.format(phase))
        try:
            with np.load(npzfile, allow_pickle=True) as f:
                return f['cache']
        except Exception:
            cache = self._load_batch(self.data, id, views)
            if data_utils.get_rank() == 0:
                np.savez(npzfile, cache=cache)
            return cache

    def select(self, file_list):
        if len(file_list[0]) == 0:
            raise FileNotFoundError
        return [[files[i] for i in self.views] for files in file_list]
    
    def find_rgb(self, name='rgb'):
        try:
            return self.select([sorted(glob.glob(path + '/{}/*.*g'.format(name))) for path in self.paths])
        except FileNotFoundError:
            try:
                return self.select([sorted(glob.glob(path + '/color/*.*g')) for path in self.paths])
            except FileNotFoundError:
                raise FileNotFoundError("CANNOT find rendered images.")
    
    def find_depth(self):
        try:
            return self.select([sorted(glob.glob(path + '/depth/*.exr')) for path in self.paths])
        except FileNotFoundError:
            raise FileNotFoundError("CANNOT find estimated depths images") 

    def find_mask(self):
        try:
            return self.select([sorted(glob.glob(path + '/mask/*')) for path in self.paths])
        except FileNotFoundError:
            raise FileNotFoundError("CANNOT find precomputed mask images")

    def find_extrinsics(self):
        try:
            return self.select([sorted(glob.glob(path + '/extrinsic/*.txt')) for path in self.paths])
        except FileNotFoundError:
            try:
                self.world2camera = False
                return self.select([sorted(glob.glob(path + '/pose/*.txt')) for path in self.paths])
            except FileNotFoundError:
                raise FileNotFoundError('world2camera or camera2world matrices not found.')   
    
    def find_intrinsics_per_view(self):
        try:
            return self.select([sorted(glob.glob(path + '/intrinsic/*.txt')) for path in self.paths])
        except FileNotFoundError:
            return None

    def summary_view_data(self, _data_per_view):
        keys = [k for k in _data_per_view if _data_per_view[k] is not None]
        num_of_objects = len(_data_per_view[keys[0]])
        for k in range(num_of_objects):
            assert len(set([len(_data_per_view[key][k]) for key in keys])) == 1, "numer of views must be consistent."
        return [list(range(len(_data_per_view[keys[0]][k]))) for k in range(num_of_objects)]

    def num_tokens(self, index):
        return self.num_view

    def _load_view(self, packed_data, view_idx):
        image, uv, ratio = data_utils.load_rgb(
            packed_data['rgb'][view_idx], 
            resolution=self.resolution,
            bg_color=self.bg_color,
            min_rgb=self.min_color)
        rgb, alpha = image[:3], image[3]  # C x H x W for RGB
        extrinsics = data_utils.load_matrix(packed_data['ext'][view_idx]) 
        extrinsics = geometry.parse_extrinsics(extrinsics, self.world2camera).astype('float32')  # this is C2W
        intrinsics = data_utils.load_intrinsics(packed_data['ixt_v'][view_idx]).astype('float32') \
            if packed_data.get('ixt_v', None) is not None else None

        z, mask = None, None
        if packed_data.get('dep', None) is not None:
            z = data_utils.load_depth(packed_data['dep'][view_idx], resolution=self.resolution)
        if packed_data.get('mask', None) is not None:
            mask = data_utils.load_mask(packed_data['mask'][view_idx], resolution=self.resolution)
            if self.apply_mask_color:   # we can also not apply mask
                rgb = rgb * mask[None, :, :] + (1 - mask[None, :, :]) * np.asarray(self.bg_color)[:, None, None]

        return {
            'path': packed_data['rgb'][view_idx],
            'view': view_idx,
            'uv': uv.reshape(2, -1), 
            'colors': rgb.reshape(3, -1), 
            'alpha': alpha.reshape(-1), 
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'depths': z.reshape(-1) if z is not None else None,
            'mask': mask.reshape(-1) if mask is not None else None,
            'size': np.array([rgb.shape[1], rgb.shape[2]] + ratio, dtype=np.float32)
        }

    def _load_batch(self, data, index, view_ids=None):
        if self.no_rgb:
            return super()._load_batch(data, index)

        if view_ids is None:
            view_ids = [next(self.data_index[index]) for _ in range(self.num_view)]
        return index, self._load_shape(data[index]), [self._load_view(data[index], view_id) for view_id in view_ids]
        
    def _load_shape(self, packed_data):
        shape_data = super()._load_shape(packed_data)
        if self.load_all_images:
            # load all image from this batch
            images = np.stack([data_utils.load_rgb(
                data, 
                resolution=self.resolution,
                bg_color=self.bg_color,
                min_rgb=self.min_color)[0].transpose(1,2,0)
            for data in packed_data['rgb']], -1)  # H x W x 4 x ALL
            shape_data['all'] = images
        return shape_data

    def __getitem__(self, index):
        if self.cache is not None:
            view_ids = [next(self.data_index[index]) for _ in range(self.num_view)]
            return copy.deepcopy(self.cache[index % self.total_num_shape][0]), \
                   copy.deepcopy(self.cache[index % self.total_num_shape][1]), \
                  [copy.deepcopy(self.cache[index % self.total_num_shape][2][i]) for i in view_ids]
        return self._load_batch(self.data, index)

    def collater(self, samples):
        results = super().collater(samples)
        if self.no_rgb:
            return results

        if results is None:
            return results

        for key in samples[0][2][0]:
            if key == 'path':
                results[key] = [[d[key] for d in s[2]] for s in samples]

            elif samples[0][2][0][key] is not None:
                results[key] = torch.from_numpy(
                    np.array([[d[key] for d in s[2]] for s in samples])
                )
                
        results['colors'] = results['colors'].transpose(2, 3)
        if results.get('full_rgb', None) is not None:
            results['full_rgb'] = results['full_rgb'].transpose(2, 3)
        return results


class VideoDataset(ShapeViewDataset):

    def find_paths(self, paths):
        return sorted(glob.glob(paths + '/transform/*.json'))
        
    def find_intrinsics(self):
        home_path = '/'.join(self.paths[0].split('/')[:-2])
        if os.path.exists(home_path + '/intrinsic.txt'):
            ixt = home_path + '/intrinsic.txt'
        elif os.path.exists(home_path + '/intrinsics.txt'):
            ixt = home_path + '/intrinsics.txt'
        else:
            return []

        return [ixt for path in self.paths]
    
    def find_global(self):
        return []

    def find_vertex(self, vertex_fname="vertex.txt"):
        return []
        
    def find_joints(self, joint_fname="joints.json"):
        home_path = '/'.join(self.paths[0].split('/')[:-2])
        if os.path.exists(home_path + '/transform'):
            jst_list = sorted(glob.glob(home_path + '/transform/*.json'))
            print(len(jst_list), len(self.paths))
            assert len(jst_list) == len(self.paths), "the number of joints did not match"
            return jst_list
        return []

    def find_textures(self):
        home_path = '/'.join(self.paths[0].split('/')[:-2])
        fdir = 'tex'
        # fdir = 'tex_pred'

        if os.path.exists(home_path + '/{}'.format(fdir)):
            tex_list = sorted(glob.glob(home_path + '/{}/*.png'.format(fdir)))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
            return tex_list
        return []
    
    def find_rgb(self, *unused):
        try:
            return self.select([sorted(glob.glob(path.replace('transform', 'rgb').replace('.json', '/*.*g'))) for path in self.paths])
        except FileNotFoundError:
            return None

    def find_intrinsics_per_view(self):
        home_path = '/'.join(self.paths[0].split('/')[:-2])
        if os.path.exists(home_path + '/intrinsic'):
            return self.select([sorted(glob.glob(home_path + '/intrinsic/*.txt'))]) * len(self.paths)
        else:
            return None

    def find_extrinsics(self):
        home_path = '/'.join(self.paths[0].split('/')[:-2])
        if os.path.exists(home_path + '/extrinsic'):
            self.world2camera = True
            path = home_path + '/extrinsic'
        elif os.path.exists(home_path + '/pose'):
            self.world2camera = False
            path = home_path + '/pose'
        else:
            raise FileNotFoundError('world2camera or camera2world matrices not found.') 

        return self.select([sorted(glob.glob(path + '/*.txt'))]) * len(self.paths)
        

class ShapeViewStreamDataset(BaseWrapperDataset):
    """
    Different from ShapeViewDataset.
    We merge all the views together into one dataset regardless of the shapes.

    ** HACK **: an alternative of the ShapeViewDataset
    """
    def __init__(self, dataset):
        super().__init__(dataset)

        self.dataset.repeat == 1
        self.dataset.num_view == 1
        self.total_num_shape = dataset.total_num_shape
        
        # reset the data_index
        self.dataset.data_index = []
        for i, d in enumerate(self.data):
            for j, _ in enumerate(d['rgb']):
                self.dataset.data_index.append((i, j))   # shape i, view j

    def __len__(self):
        return len(self.dataset.data_index)

    def ordered_indices(self):
        return np.arange(len(self))

    @property
    def cache(self):
        return self.dataset.cache
    
    @property
    def data(self):
        return self.dataset.data

    def _load_batch(self, data, shape_id, view_ids):
        return shape_id, self.dataset._load_shape(data[shape_id]), [self.dataset._load_view(data[shape_id], view_id) for view_id in view_ids]

    def __getitem__(self, index):
        shape_id, view_id = self.dataset.data_index[index]
        if self.cache is not None:
            return copy.deepcopy(self.cache[shape_id % self.total_num_shape][0]), \
                   copy.deepcopy(self.cache[shape_id % self.total_num_shape][1]), \
                  [copy.deepcopy(self.cache[shape_id % self.total_num_shape][2][view_id])]
        return self._load_batch(self.data, shape_id, [view_id])

    def _load_binary(self, id, views, phase='train'):
        root = os.path.dirname(self.data[id]['ixt'])
        npzfile = os.path.join(root, '{}.npz'.format(phase))
        try:
            with np.load(npzfile, allow_pickle=True) as f:
                return f['cache']
        except Exception:
            caches = [self._load_batch(self.data, id, view_id) for view_id in views]
            cache = [caches[0][0], caches[0][1], [caches[i][2][0] for i in range(len(views))]]
           
            if data_utils.get_rank() == 0:
                np.savez(npzfile, cache=cache)
            return cache


class SampledPixelDataset(BaseWrapperDataset):
    """
    A wrapper dataset, which split rendered images into pixels
    """

    def __init__(self, 
        dataset, 
        num_sample=None, 
        sampling_on_mask=1.0, 
        sampling_on_bbox=False, 
        sampling_at_center=1.0,
        resolution=512, 
        patch_size=1):
        
        super().__init__(dataset)
        self.num_sample = num_sample
        self.sampling_on_mask = sampling_on_mask
        self.sampling_on_bbox = sampling_on_bbox
        self.sampling_at_center = sampling_at_center
        self.patch_size = patch_size
        self.res = resolution

    def __getitem__(self, index):
        index, data_per_shape, data_per_view = self.dataset[index]

        # sample pixels from the original images
        sample_index = [
            data_utils.sample_pixel_from_image(
                data['alpha'].shape[-1], 
                self.num_sample, 
                data.get('mask', None) 
                    if data.get('mask', None) is not None 
                    else data.get('alpha', None),
                self.sampling_on_mask,
                self.sampling_on_bbox,
                self.sampling_at_center,
                width=int(data['size'][1]),
                patch_size=self.patch_size)
            for data in data_per_view
        ]
        
        for i, data in enumerate(data_per_view):
            data_per_view[i]['full_rgb'] = copy.deepcopy(data['colors'])
            for key in data:
                if data[key] is not None \
                    and (key != 'extrinsics' and key != 'view' and key != 'full_rgb') \
                    and data[key].shape[-1] > self.num_sample:

                    if len(data[key].shape) == 2:
                        data_per_view[i][key] = data[key][:, sample_index[i]] 
                    else:
                        data_per_view[i][key] = data[key][sample_index[i]]
            data_per_view[i]['index'] = sample_index[i]
        return index, data_per_shape, data_per_view

    def num_tokens(self, index):
        return self.dataset.num_view * self.num_sample 


class WorldCoordDataset(BaseWrapperDataset):
    """
    A wrapper dataset. transform UV space into World space
    """
    def __getitem__(self, index):
        index, data_per_shape, data_per_view = self.dataset[index]

        def camera2world(data):
            inv_RT = data['extrinsics']
            intrinsics = data_per_shape['intrinsics']

            # get camera center (XYZ)
            ray_start = inv_RT[:3, 3]

            # get points at a random depth (=1)
            ray_dir = geometry.get_ray_direction(
                ray_start, data['uv'], intrinsics, inv_RT, 1
            )
            
            # here we still keep the original data for tracking purpose
            data.update({
                'ray_start': ray_start,
                'ray_dir': ray_dir,
            })
            return data

        return index, data_per_shape, [camera2world(data) for data in data_per_view]
        
    def collater(self, samples):
        results = self.dataset.collater(samples)
        if results is None:
            return results

        results['ray_start'] = results['ray_start'].unsqueeze(-2)
        results['ray_dir'] = results['ray_dir'].transpose(2, 3)
        results['colors'] = results['colors'].transpose(2, 3)
        if results.get('full_rgb', None) is not None:
            results['full_rgb'] = results['full_rgb'].transpose(2, 3)
        return results


class InfiniteDataset(BaseWrapperDataset):
    """
    A wrapper dataset which supports infnite sampling from a dataset.
    No epochs in this case.
    """
    def __init__(self, dataset, max_len=1000000):
        super().__init__(dataset)
        self.MAXLEN = max_len

    def __len__(self):
        return self.MAXLEN

    def ordered_indices(self):
        return np.arange(self.MAXLEN)

    def __getitem__(self, index):
        actual_length = len(self.dataset)
        return self.dataset[index % actual_length]