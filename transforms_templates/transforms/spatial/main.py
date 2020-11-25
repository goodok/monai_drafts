import numpy as np
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union, Collection

from monai.transforms import Randomizable, MapTransform
from monai.transforms.spatial.array import Flip
from monai.config import KeysCollection

from ..utils.main import is_mesh, is_image_2d, is_points
from .points import crop_rect

class RandFlipX(MapTransform, Randomizable):
    """
    Usage in config:
        - _target_: tt.RandFlip
        keys: ['mesh', 'image2d']
        prob: 0.5
    """

    # TODO:
    # - work with kind of 'image2d' (top, left, right)

    # Random:
    # https://github.com/Project-MONAI/MONAI/blob/master/monai/transforms/spatial/dictionary.py#L232
    # Flip/Flipd
    # https://github.com/Project-MONAI/MONAI/blob/master/monai/transforms/spatial/dictionary.py#L657
    def __init__(
        self,
        keys: KeysCollection,
        image_spatial_axis: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
        """
        super().__init__(keys)

        self.prob = min(max(prob, 0.0), 1.0)

        self.image_flipper = Flip(spatial_axis=image_spatial_axis)

        self._do_transform = False

    def randomize(self):
        self._do_transform = self.R.random() < self.prob

    def __call__(self, d):
        self.randomize()
        if not self._do_transform:
            return d

        m = self.calc_affine_matrix()

        for key in self.keys:
            item = d[key]
            if is_mesh(item):
                # inplace
                item.aplly_affine(m)                
                item.data.invert() # Invert the mesh in-place by reversing the winding of every face and negating normals
                d[key] = item
            elif is_image_2d(item):
                # TODO: implement
                d[key] = self.image_flipper(item)
            else:
                raise NotImplementedError

        return d

    def calc_affine_matrix(self):
        # For mesh or point cloud
        m = np.diag([-1, 1, 1, 1]).astype(np.float32)
        return m


class RandRotateXY(MapTransform, Randomizable):
    """
    Usage in config:
        - _target_: tt.RandRotateXY
        keys: ['mesh']
        prob: 0.5
    """

    # Random:
    # https://github.com/Project-MONAI/MONAI/blob/master/monai/transforms/spatial/dictionary.py#L232
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        #spatial_axes: Tuple[int, int] = (0, 1),
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        super().__init__(keys)

        self.prob = min(max(prob, 0.0), 1.0)
        #self.spatial_axes = spatial_axes

        self._do_transform = False

    def randomize(self):
        self._do_transform = self.R.random() < self.prob
        self._theta = self.R.rand() * 2 * np.pi

    def __call__(self, d):
        self.randomize()
        if not self._do_transform:
            return d

        m = self.calc_affine_matrix()

        for key in self.keys:
            item = d[key]
            # inplace
            item.aplly_affine(m)

            d[key] = item

        d['_theta'] = self._theta

        return d

    def calc_affine_matrix(self):
        theta = self._theta
        Q = np.array(
            [[np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]],
            dtype=np.float32)

        # transformation matrix
        m = np.eye(4, dtype='float32')
        m[:2, :2] = Q
        return m


class CropRect(MapTransform):
    """
    Usage in config:
        - _target_: tt.CropRect
        keys: ['points']
        sizes: [0, -40, -3, 70.4, 40, 1] # xyzxyz
    """

    def __init__(
        self,
        keys: KeysCollection,
        sizes: Collection[float],
        prob: float = 0.1,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
        """
        super().__init__(keys)


        self.sizes = sizes

    def __call__(self, d):

        for key in self.keys:
            item = d[key]
            if is_points(item):
                crop_rect(item, self.sizes)
                d[key] = item
            else:
                raise NotImplementedError
        return d
