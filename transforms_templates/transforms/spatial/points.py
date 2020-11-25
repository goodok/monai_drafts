import numpy as np

from transforms_templates.items.items import PointsItem
from transforms_templates.transforms.utils.main import is_points

# https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/core/data_transform/transforms.py#L823

def crop_rect(item: PointsItem, sizes):
    """
    Args:
        sizes: [6] list/tuple or array, float. indicate voxel
            sizes. format: xyzxyz, minmax
    """
    sizes = np.array(sizes, dtype=np.float32)
    data = item.data
    points = data['points']
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ind_x = (x >= sizes[0]) & (x <= sizes[3])
    ind_y = (y >= sizes[1]) & (x <= sizes[4])
    ind_z = (z >= sizes[2]) & (z <= sizes[5])
    ind = ind_x & ind_y & ind_z

    points_new = points[ind]

    data['points'] = points_new

    # work with features and labels0
    keys = ['features']
    if 'labels_keys' in data:
        keys += data['labels_keys']
    for key in keys:
        if key in data:
            data[key] = data[key][ind]

    # TODO: save indices if needed
    return item