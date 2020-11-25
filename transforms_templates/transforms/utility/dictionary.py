from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union
import numpy as np
import torch


from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection
from monai.transforms import Transpose

class AddSubItemsd(MapTransform):
    """
    Get specified value of items from data dictionary, assuming that the item is nested dictionary also
    For example, 
    data dictionary = {'target': array; 'input': image, 'meta': {'camera':{'distance': 2.0}, 'pos':[0,1,2]}} 
    AddSubItems(keys: ['meta'], subkeys = ['camera.pos'], new_keys = ['camera_distance']) 
    add new 'camera_distance' key to data dictionary with value 2.0
    """

    def __init__(self, keys: KeysCollection, subkeys:KeysCollection, new_keys:KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        
        assert len(keys) == len(subkeys), f'new columns: {keys}; columns: {subkeys}'
        assert len(keys) == len(new_keys), f'new columns: {keys}; columns: {new_keys}'
        
        self.subkeys = subkeys
        self.new_keys = new_keys      

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)        
        for key, subkey, new_key in zip(self.keys, self.subkeys, self.new_keys):            
            d[new_key] = np.array(d[key][subkey])
        return d

AddSubItemsD = AddSubItemsDict = AddSubItemsd