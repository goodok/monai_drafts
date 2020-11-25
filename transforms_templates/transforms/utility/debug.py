from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection

from transforms_templates.utils.log import log
from ..utils.main import is_mesh, is_points, is_image_2d

class DebugDict(MapTransform):
    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)
        self._counter = 0
        self._is_debug = True

    def __call__(self, data):
        d = dict(data)
        print(f'Debug {self._counter}')
        print(f'  keys: {list(d.keys())}')
        print(f'  self.keys: {self.keys}')
        if len(self.keys) == 1 and self.keys[0] is None:
            log(d)
        else:
            for key in self.keys:
                item = d[key]
                if is_mesh(item):
                    print(f"'{key}':")
                    item.describe()
                elif is_image_2d(item):
                    log(f"'{key}'", item)
                elif is_points(item):
                    print(f"'{key}':")
                    item.describe()
                else:
                    print(f"'{key}':")
                    log(item)
        print()
        return d


DebugD = DebugDict
Debugd = DebugDict