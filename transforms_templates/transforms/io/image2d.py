from pathlib import Path
import json

from monai.transforms import LoadDatad, LoadPNG
from monai.config import KeysCollection


class LoadPNGandJSONd(LoadDatad):
    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
    ) -> None:
        """
        """
        loader = LoadPNG()
        super().__init__(keys, loader, meta_key_postfix, overwriting)

    def load_info(self, fn_original, fn_info=None):
        # load json info to meta
        res = None
        if fn_info is not None:
            fn = fn_info
        else:
            fn = Path(fn_original).with_suffix('.info')
        if fn.exists():
            with open(fn) as f:
                res = json.load(f)
        return res

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key in self.keys:
            data = self.loader(d[key])
            fn_original = d[key]
            fn_info_key = f"{key}_{self.meta_key_postfix}"
            fn_info = d.get(fn_info_key, None)

            meta_data_2 = self.load_info(fn_original, fn_info)
            assert meta_data_2 is not None, f" File {fn_original} or {fn_info} is not found."
            assert isinstance(data, (tuple, list)), "loader must return a tuple or list."
            d[key] = data[0]
            assert isinstance(data[1], dict), "metadata must be a dict."
            key_to_add = f"{key}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(f"Meta data with key {key_to_add} already exists and overwriting=False.")
            d[key_to_add] = data[1]
            if meta_data_2 is not None:
                d[key_to_add].update(meta_data_2)
        return d
