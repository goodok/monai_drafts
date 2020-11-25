import numpy as np
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection

from ..utils.main import is_mesh, is_image_2d
from transforms_templates.items.items import PointsItem

class SamplePoints(MapTransform):
    """
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_points: int = 50000,
        replace: bool =True,
    ) -> None:
        """
        """
        super().__init__(keys)

        self.num_points = num_points
        self.replace = replace

    def __call__(self, d):

        for key in self.keys:
            item = d[key]
            # TODO: check inplace
            item = _sample_points(item, num_points=self.num_points, replace=self.replace)
            # d[key] = item
        return d


def _sample_points(x: PointsItem, num_points=50000, replace=True):

    d = x.data
    n = num_points

    num_points_was = len(d['points'])

    if n > 0:
        n = np.min([n, num_points_was])
        if replace:
            indices = np.random.randint(num_points_was, size=n)
        else:
            indices = np.random.choice(num_points_was, size=n, replace=False)
        # TODO: work with features
        keys = ['points', 'features']
        if 'labels_keys' in d:
            keys += d['labels_keys']
        for k in keys:
            if k in d:
                assert len(d[k]) == num_points_was
                d[k] = d[k][indices]
    return x

