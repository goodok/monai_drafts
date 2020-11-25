
import numpy as np
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import torch

from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection

from ..utils.main import is_mesh, is_image_2d, is_points
from transforms_templates.items.items import PointsItem

# TODO:
#-  new_keys_prefixes: KeysCollection = None,



class ToSparseVoxels(MapTransform):
    """
    """

    # TODO:
    # - work with features
    # - add_local_pos (deltas) parameter

    def __init__(
        self,
        keys: KeysCollection,
        voxel_size,
        coords_range,
        new_keys_prefixes: KeysCollection = None,
        add_local_pos=False,

    ) -> None:
        """
        Parameters
        ==========

            # points : [N, ndim] float tensor. points[:, :3]
            #   contain xyz points
            #    and points[:, 3:] contain other information like reflectivity
            voxel_size : [3] list/tuple or array, float. xyz, indicate voxel
                size
            coords_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            add_local_pos: bool
                add local position in voxels to featues
        """
        super().__init__(keys)

        if new_keys_prefixes is None:
            new_keys_prefixes = keys
        assert len(new_keys_prefixes) == len(keys)

        self.new_keys_prefixes = new_keys_prefixes
        self.voxel_size = voxel_size
        self.coords_range = coords_range
        self.add_local_pos = add_local_pos

    def __call__(self, d):

        for key, prefix in zip(self.keys, self.new_keys_prefixes):
            if prefix:
                prefix = prefix + "_"

            item = d[key]
            data = item.data
            assert is_points(item)
            points = data['points']
            res = _points_to_sparse(
                points,
                voxel_size=self.voxel_size,
                coords_range=self.coords_range,
                add_local_pos=self.add_local_pos,
                )
            del d[key]
            d[prefix + 'coords'] = res['coords']

            features = data['features']
            if self.add_local_pos:
                data['features_original_keys'] += ['dx', 'dy', 'dz']
                features = np.hstack([features, res['dxdydz']])
            d[prefix + 'features'] = features

            for label_key in data['labels_keys']:
                d[prefix + label_key] = data[label_key]

            d[prefix + 'features_original_keys'] = data['features_original_keys']
            d[prefix + 'labels_keys']= data['labels_keys']
            
        return d

# https://github.com/goodok/fastai_sparse/blob/master/fastai_sparse/transforms/convert.py#L96
def _points_to_sparse(
    points,
    voxel_size,
    coords_range,
    add_local_pos=False):

    sizes = np.array(coords_range, dtype=np.float32)
    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = np.array(voxel_size, dtype=np.float32)

    grid_size = (sizes[3:] - sizes[:3]) / voxel_size
    
    coords_0 = sizes[:3]
    coords = (points - coords_0) / voxel_size
    coords = np.floor(coords).astype(np.int64)

    if (coords < 0).any():
        print(points.min())
        print(coords_0)
        print((points - coords_0).min())
        print( ((points - coords_0) / voxel_size).min())
        raise Exception(f'coords.min()={coords.min()} ')
    
   
    res = {'coords': coords, 'grid_size': grid_size}

    dxdydz = None
    if add_local_pos:
        dxdydz = np.array(points - coords - 0.5, dtype=np.float32)
        res['dxdydz'] = dxdydz

    return res

    

# # Based on old version
# # https://github.com/traveller59/second.pytorch/blob/v1.0/second/core/voxel_generator.py

# def _points_to_sparse_voxels_1(
#     points,
#     voxel_size,
#     coors_range,
#     max_points=35,
#     max_voxels=20000):
#     """convert kitti points(N, >=3) to voxels.
#     Args:
#         points: [N, ndim] float tensor. points[:, :3] contain xyz points
#             and points[:, 3:] contain other information like reflectivity
#         voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
#             size
#         coors_range: [6] list/tuple or array, float. indicate voxel
#             range. format: xyzxyz, minmax
#         max_points: int. indicate maximum points contained in a voxel. if
#             max_points=-1, it means using dynamic_voxelize
#         max_voxels: int. indicate maximum voxels this function create.
#             for second, 20000 is a good choice. Users should shuffle points
#             before call this function because max_voxels may drop points.
#     Returns:
#         voxels: [M, max_points, ndim] float tensor. only contain points
#                 and returned when max_points != -1.
#         coordinates: [M, 3] int32 tensor, always returned.
#         num_points_per_voxel: [M] int32 tensor. Only returned when
#             max_points != -1.
#     """

#     from second.core.point_cloud.point_cloud_ops import points_to_voxel


#     point_cloud_range = np.array(coors_range, dtype=np.float32)
#     # [0, -40, -3, 70.4, 40, 1]
#     voxel_size = np.array(voxel_size, dtype=np.float32)

#     grid_size = (
#             point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
#     grid_size = np.round(grid_size).astype(np.int64)   # TODO: round twice ?
#     voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())

#     _coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)

#     return points_to_voxel(voxel_size, point_cloud_range, _coor_to_voxelidx, max_points, True, max_voxels)



# # https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/voxelize.py#L13
# # CUDA optimized
# def _points_to_sparse_voxels_2(
#     points,
#     voxel_size,
#     coors_range,
#     max_points=35,
#     max_voxels=20000):
#     """convert kitti points(N, >=3) to voxels.
#     Args:
#         points: [N, ndim] float tensor. points[:, :3] contain xyz points
#             and points[:, 3:] contain other information like reflectivity
#         voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
#             size
#         coors_range: [6] list/tuple or array, float. indicate voxel
#             range. format: xyzxyz, minmax
#         max_points: int. indicate maximum points contained in a voxel. if
#             max_points=-1, it means using dynamic_voxelize
#         max_voxels: int. indicate maximum voxels this function create.
#             for second, 20000 is a good choice. Users should shuffle points
#             before call this function because max_voxels may drop points.
#     Returns:
#         voxels: [M, max_points, ndim] float tensor. only contain points
#                 and returned when max_points != -1.
#         coordinates: [M, 3] int32 tensor, always returned.
#         num_points_per_voxel: [M] int32 tensor. Only returned when
#             max_points != -1.
#     """
#     if max_points == -1 or max_voxels == -1:
#         coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
#         dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
#         return coors
#     else:
#         voxels = points.new_zeros(
#             size=(max_voxels, max_points, points.size(1)))
#         coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
#         num_points_per_voxel = points.new_zeros(
#             size=(max_voxels, ), dtype=torch.int)
#         voxel_num = hard_voxelize(points, voxels, coors,
#                                     num_points_per_voxel, voxel_size,
#                                     coors_range, max_points, max_voxels, 3)
#         # select the valid voxels
#         voxels_out = voxels[:voxel_num]
#         coors_out = coors[:voxel_num]
#         num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
#         return voxels_out, coors_out, num_points_per_voxel_out