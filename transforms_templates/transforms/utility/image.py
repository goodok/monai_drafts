from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union
import numpy as np
import torch


from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection
from monai.transforms import Transpose

class Transposed(MapTransform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, keys: KeysCollection, indices: Optional[Sequence[int]]) -> None:
        super().__init__(keys)
        self.converter = Transpose(indices=indices)

    def __call__(self,  data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d