import torch
import numpy as np

class Collater():
    """
    For use in configs (hydra)
    """
    def __init__(self,
                 as_list = ['id', 'num_points'],
                 as_stack = ['image2d'],  # tensor [(C, H, W), (C, H, W), ] ---> [B, C, H, W, ...]
                 as_pack=['sp_features', 'sp_shape_id'],
                 as_pack_with_index=['sp_coords'],  # for SparseConNet
                 num_points_source_key='sp_coords',
                 ):
        self.as_list = as_list
        self.as_stack = as_stack
        self.as_pack = as_pack
        self.as_pack_with_index = as_pack_with_index
        self.num_points_source_key = num_points_source_key
        
    # __call__ ?
    def collate(self, examples):
        res = {}
    
        for key in self.as_pack_with_index:
            a = [d[key] for d in examples]
            ones = torch.from_numpy(np.vstack([idx * np.ones((x.shape[0], 1), dtype="int64") for idx, x in enumerate(a)]))
            a = torch.cat(a, dim=0)
            a = torch.cat([a, ones], dim=1)
            res[key] = a

        for key in self.as_pack:
            a = [d[key] for d in examples]
            a = torch.cat(a, dim=0)
            res[key] = a

        if "num_points" in self.as_list:
            num_points = [len(d[self.num_points_source_key]) for d in examples]
            res["num_points"] = num_points

        return res

    def num_points_of_example(self, example, key=['sp_coords']):
        # TODO: as transform which fill aditioanal key 'num_points'
        return len(example[key])
