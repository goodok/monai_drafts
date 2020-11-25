import numpy as np
from transforms_templates.items.items import MeshItem, PointsItem


def mesh_keys(o):
    return [key for key, item in o.items() if is_mesh(o)]

def is_mesh(item):
    return isinstance(item, MeshItem)

def is_points(item):
    return isinstance(item, PointsItem)

def is_image_2d(item):
    return isinstance(item, np.ndarray) and (item.ndim == 3)