import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Hashable, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import KeysCollection
from monai.transforms.utils import apply_transform
from monai.utils import ensure_tuple, get_seed

from monai.transforms import Compose as Compose_Base

from .utility.debug import DebugDict

class Compose(Compose_Base):

    def __call__(self, input_, debug=False):
        debug_counter = 1
        for _transform in self.transforms:
            if getattr(_transform, '_is_debug', False):
                if debug:
                    _transform._counter = debug_counter
                    #input_ = apply_transform(_transform, input_)
                    apply_transform(_transform, input_)
                    debug_counter += 1
                continue

            input_ = apply_transform(_transform, input_)
        return input_